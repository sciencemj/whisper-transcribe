"""Desktop GUI (Flet) for turning meeting recordings into minutes.

Everything runs on this machine: Whisper for speech-to-text and a local
LLM server (Ollama / OpenAI-compatible) for the summary. Flet's desktop
client talks to this Python process over a local socket only; it is
downloaded once on first launch and cached afterwards.
"""

import os
import re
import subprocess
import sys
import traceback
from pathlib import Path

import flet as ft

from . import llm
from .minutes import output_path
from .pipeline import TRANSCRIPT_SUFFIXES, MeetingOptions, run_meeting

WHISPER_MODELS = ["tiny", "base", "small", "medium", "large", "turbo"]
LANGUAGES = {"ko": "한국어", "en": "영어", "ja": "일본어", "zh": "중국어", "": "자동 감지"}
DEVICES = {"auto": "자동", "cuda": "CUDA", "mps": "Apple (MPS)", "cpu": "CPU"}
SOURCE_EXTENSIONS = ["m4a", "mp3", "wav", "aac", "flac", "ogg", "webm", "mp4", "mov", "txt", "md"]


def _open_path(path: Path) -> None:
    if sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    elif os.name == "nt":
        os.startfile(path)  # type: ignore[attr-defined]
    else:
        subprocess.Popen(["xdg-open", str(path)])


def _preview_markdown(text: str) -> str:
    # Flet's Markdown doesn't render HTML; unwrap the collapsible transcript.
    return re.sub(r"</?details>|<summary>.*?</summary>", "", text)


def _dropdown(label: str, options: dict[str, str], value: str, **kwargs) -> ft.Dropdown:
    return ft.Dropdown(
        label=label,
        value=value,
        options=[ft.DropdownOption(key=k, text=v) for k, v in options.items()],
        dense=True,
        **kwargs,
    )


def _section(title: str, icon, *controls: ft.Control) -> ft.Card:
    return ft.Card(
        content=ft.Container(
            padding=16,
            content=ft.Column(
                spacing=12,
                controls=[
                    ft.Row([ft.Icon(icon, size=18, color=ft.Colors.PRIMARY), ft.Text(title, weight=ft.FontWeight.W_600)]),
                    *controls,
                ],
            ),
        ),
    )


class MeetingApp:
    def __init__(self, page: ft.Page):
        self.page = page
        self.model_cache: dict[tuple, object] = {}
        self.result_path: Path | None = None
        self.busy = False

        page.title = "회의록 생성기"
        page.theme_mode = ft.ThemeMode.SYSTEM
        page.theme = ft.Theme(color_scheme_seed=ft.Colors.INDIGO)
        page.dark_theme = ft.Theme(color_scheme_seed=ft.Colors.INDIGO)
        page.padding = 0
        page.window.width = 1180
        page.window.height = 820
        page.window.min_width = 900
        page.window.min_height = 640

        self.picker = ft.FilePicker()

        # --- inputs ----------------------------------------------------
        self.source = ft.TextField(label="회의 녹음 또는 녹취록", dense=True, expand=True, on_change=self._on_source_change)
        self.out_dir = ft.TextField(label="저장 폴더", value=str(Path.home() / "Documents"), dense=True, expand=True)
        self.title = ft.TextField(label="회의 제목", hint_text="비워 두면 파일 이름 사용", dense=True)
        self.attendees = ft.TextField(label="참석자", hint_text="예: 김철수, 이영희", dense=True)

        self.whisper_model = _dropdown("모델", {m: m for m in WHISPER_MODELS}, "turbo", expand=True)
        self.language = _dropdown("언어", LANGUAGES, "ko", expand=True)
        self.device = _dropdown("장치", DEVICES, "auto", expand=True)

        self.summarize = ft.Switch(label="로컬 LLM으로 안건·결정사항·할 일 정리", value=True, on_change=self._on_summarize)
        self.backend = _dropdown(
            "서버", {"ollama": "Ollama", "openai": "OpenAI 호환"}, "ollama", width=150, on_select=self._on_backend
        )
        self.llm_url = ft.TextField(label="URL", value=llm.DEFAULT_URLS["ollama"], dense=True, expand=True)
        self.llm_model = ft.Dropdown(
            label="LLM 모델",
            value=llm.DEFAULT_MODEL,
            options=[ft.DropdownOption(key=llm.DEFAULT_MODEL)],
            editable=True,
            dense=True,
            expand=True,
        )
        self.llm_settings = ft.Column(
            spacing=12,
            controls=[
                ft.Row([self.backend, self.llm_url]),
                ft.Row(
                    [
                        self.llm_model,
                        ft.IconButton(ft.Icons.REFRESH, tooltip="설치된 로컬 모델 불러오기", on_click=self._refresh_models),
                    ]
                ),
            ],
        )

        # --- run / output ----------------------------------------------
        self.run_btn = ft.FilledButton(
            "회의록 생성", icon=ft.Icons.AUTO_AWESOME, on_click=self._start, height=44
        )
        self.progress = ft.ProgressBar(value=0, expand=True)
        self.status = ft.Text("대기 중", color=ft.Colors.ON_SURFACE_VARIANT)
        self.preview = ft.Markdown(
            "녹음 파일을 선택하고 **회의록 생성**을 누르세요.",
            selectable=True,
            extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
        )
        self.log_view = ft.ListView(spacing=2, auto_scroll=True, expand=True)
        self.open_btn = ft.OutlinedButton("파일 열기", icon=ft.Icons.DESCRIPTION, disabled=True, on_click=self._open_result)
        self.copy_btn = ft.OutlinedButton("복사", icon=ft.Icons.COPY, disabled=True, on_click=self._copy_result)
        self.folder_btn = ft.OutlinedButton(
            "폴더 열기", icon=ft.Icons.FOLDER_OPEN, on_click=lambda _: _open_path(Path(self.out_dir.value).expanduser())
        )
        self.tabs = ft.Tabs(
            length=2,
            expand=True,
            content=ft.Column(
                expand=True,
                controls=[
                    ft.TabBar(tabs=[ft.Tab(label="결과", icon=ft.Icons.ARTICLE), ft.Tab(label="로그", icon=ft.Icons.TERMINAL)]),
                    ft.TabBarView(
                        expand=True,
                        controls=[
                            ft.Container(
                                padding=16,
                                content=ft.Column([self.preview], scroll=ft.ScrollMode.AUTO, expand=True),
                            ),
                            ft.Container(padding=16, content=self.log_view),
                        ],
                    ),
                ],
            ),
        )

        page.add(self._layout())

    def _layout(self) -> ft.Control:
        header = ft.Container(
            padding=ft.Padding.symmetric(horizontal=24, vertical=14),
            bgcolor=ft.Colors.SURFACE_CONTAINER,
            content=ft.Row(
                [
                    ft.Icon(ft.Icons.RECORD_VOICE_OVER, color=ft.Colors.PRIMARY),
                    ft.Text("회의록 생성기", size=20, weight=ft.FontWeight.BOLD),
                    ft.Container(expand=True),
                    ft.Container(
                        padding=ft.Padding.symmetric(horizontal=12, vertical=6),
                        border_radius=16,
                        bgcolor=ft.Colors.PRIMARY_CONTAINER,
                        tooltip="음성 인식과 AI 정리가 모두 이 컴퓨터에서 실행됩니다.",
                        content=ft.Row(
                            tight=True,
                            spacing=6,
                            controls=[
                                ft.Icon(ft.Icons.LOCK, size=16, color=ft.Colors.ON_PRIMARY_CONTAINER),
                                ft.Text("완전 로컬 처리", color=ft.Colors.ON_PRIMARY_CONTAINER, weight=ft.FontWeight.W_500),
                            ],
                        ),
                    ),
                ]
            ),
        )
        settings = ft.Container(
            width=420,
            padding=ft.Padding.only(left=16, top=16, bottom=16),
            content=ft.Column(
                scroll=ft.ScrollMode.AUTO,
                spacing=8,
                controls=[
                    _section(
                        "파일",
                        ft.Icons.AUDIO_FILE,
                        ft.Row([self.source, ft.IconButton(ft.Icons.FOLDER_OPEN, tooltip="찾아보기", on_click=self._pick_source)]),
                        ft.Row([self.out_dir, ft.IconButton(ft.Icons.DRIVE_FOLDER_UPLOAD, tooltip="찾아보기", on_click=self._pick_out_dir)]),
                    ),
                    _section("회의 정보 (선택)", ft.Icons.EVENT_NOTE, self.title, self.attendees),
                    _section(
                        "음성 인식 · Whisper",
                        ft.Icons.GRAPHIC_EQ,
                        ft.Row([self.whisper_model, self.language, self.device]),
                    ),
                    _section("로컬 AI 정리", ft.Icons.PSYCHOLOGY, self.summarize, self.llm_settings),
                ],
            ),
        )
        output = ft.Container(
            expand=True,
            padding=16,
            content=ft.Column(
                expand=True,
                controls=[
                    ft.Row([self.run_btn, ft.Column([self.progress, self.status], spacing=4, expand=True)], spacing=16),
                    ft.Card(expand=True, content=self.tabs),
                    ft.Row([self.copy_btn, self.open_btn, self.folder_btn], alignment=ft.MainAxisAlignment.END),
                ],
            ),
        )
        return ft.Column(
            expand=True,
            spacing=0,
            controls=[header, ft.Row([settings, output], expand=True, vertical_alignment=ft.CrossAxisAlignment.STRETCH)],
        )

    # ---- helpers --------------------------------------------------------
    def _snack(self, message: str, error: bool = False) -> None:
        self.page.show_dialog(
            ft.SnackBar(
                ft.Text(message),
                bgcolor=ft.Colors.ERROR_CONTAINER if error else None,
                duration=ft.Duration(seconds=8 if error else 4),
            )
        )

    def _log(self, msg: str) -> None:
        self.log_view.controls.append(ft.Text(msg.rstrip(), font_family="monospace", size=12, selectable=True))
        if msg.startswith(("summarizing", "writing final")):
            self.status.value = "로컬 AI로 정리 중…"
            self.progress.value = None  # indeterminate while the LLM works
        self.page.update()

    def _set_progress(self, frac: float) -> None:
        self.progress.value = frac
        self.status.value = f"음성 인식 중… {frac:.0%}"
        self.page.update()

    def _make_llm(self) -> llm.LocalLLM:
        return llm.LocalLLM(
            model=(self.llm_model.value or "").strip(),
            backend=self.backend.value,
            url=(self.llm_url.value or "").strip() or None,
        )

    # ---- event handlers -------------------------------------------------
    async def _pick_source(self, _e) -> None:
        files = await self.picker.pick_files(
            dialog_title="회의 녹음 선택",
            file_type=ft.FilePickerFileType.CUSTOM,
            allowed_extensions=SOURCE_EXTENSIONS,
        )
        if files:
            self.source.value = files[0].path
            self._on_source_change(None)
            self.page.update()

    async def _pick_out_dir(self, _e) -> None:
        path = await self.picker.get_directory_path(dialog_title="저장 폴더 선택", initial_directory=self.out_dir.value)
        if path:
            self.out_dir.value = path
            self.page.update()

    def _on_source_change(self, _e) -> None:
        if self.source.value and not self.title.value:
            self.title.hint_text = f"{Path(self.source.value).stem} 회의록"

    def _on_summarize(self, _e) -> None:
        self.llm_settings.disabled = not self.summarize.value
        self.page.update()

    def _on_backend(self, _e) -> None:
        self.llm_url.value = llm.DEFAULT_URLS[self.backend.value]
        self.page.update()

    def _refresh_models(self, _e) -> None:
        try:
            models = self._make_llm().list_models()
        except llm.LLMError as e:
            self._snack(str(e), error=True)
            return
        if not models:
            self._snack(
                f"로컬 모델 목록을 가져오지 못했습니다. Ollama 실행 여부를 확인하세요 (ollama serve / ollama pull {llm.DEFAULT_MODEL}).",
                error=True,
            )
            return
        self.llm_model.options = [ft.DropdownOption(key=m) for m in models]
        if self.llm_model.value not in models:
            self.llm_model.value = models[0]
        self._snack(f"로컬 모델 {len(models)}개를 찾았습니다.")
        self.page.update()

    def _start(self, _e) -> None:
        if self.busy:
            return
        source = Path((self.source.value or "").strip()).expanduser()
        if not source.is_file():
            self._snack("회의 녹음 파일을 선택하세요.", error=True)
            return
        if source.suffix.lower() in TRANSCRIPT_SUFFIXES and not self.summarize.value:
            self._snack("녹취록 파일은 '로컬 AI 정리'를 켠 경우에만 사용할 수 있습니다.", error=True)
            return
        try:
            client = self._make_llm() if self.summarize.value else None
        except llm.LLMError as e:
            self._snack(str(e), error=True)
            return

        opts = MeetingOptions(
            whisper_model=self.whisper_model.value,
            device=None if self.device.value == "auto" else self.device.value,
            language=self.language.value,
            title=(self.title.value or "").strip() or None,
            attendees=(self.attendees.value or "").strip() or None,
            out_dir=Path(self.out_dir.value).expanduser(),
            summarize=self.summarize.value,
            llm_client=client,
        )

        self.busy = True
        self.run_btn.disabled = True
        self.open_btn.disabled = self.copy_btn.disabled = True
        self.progress.value = 0
        self.status.value = "준비 중…"
        self.log_view.controls.clear()
        self.tabs.selected_index = 1
        self.page.update()
        # run_thread carries the Flet session context, so page.update() from the
        # worker reaches the client (a bare threading.Thread does not).
        self.page.run_thread(self._work, source, opts)

    def _work(self, source: Path, opts: MeetingOptions) -> None:
        from . import core

        try:
            model = None
            if source.suffix.lower() not in TRANSCRIPT_SUFFIXES:
                key = (opts.whisper_model, opts.device)
                if key not in self.model_cache:
                    self.status.value = "Whisper 모델 불러오는 중…"
                    self.progress.value = None
                    self.page.update()
                    self.model_cache = {key: core.load_model(opts.whisper_model, opts.device, log=self._log)}
                model = self.model_cache[key]
                self.progress.value = 0
            out = run_meeting(source, opts, log=self._log, progress=self._set_progress, model=model)
            self._finish(out, "완료")
        except llm.LLMError as e:
            out = output_path(source, opts.out_dir)
            if out.is_file():
                self._finish(out, "녹취록만 저장됨")
            self._fail(f"AI 정리 실패 (녹취록은 저장됨): {e}")
        except Exception as e:  # surface anything else in the GUI instead of dying silently
            self._log(traceback.format_exc())
            self._fail(f"{type(e).__name__}: {e}")

    def _finish(self, out: Path, status: str) -> None:
        self.busy = False
        self.result_path = out
        self.preview.value = _preview_markdown(out.read_text(encoding="utf-8"))
        self.progress.value = 1
        self.status.value = f"{status} · {out.name}"
        self.run_btn.disabled = False
        self.open_btn.disabled = self.copy_btn.disabled = False
        self.tabs.selected_index = 0
        self.page.update()

    def _fail(self, msg: str) -> None:
        self.busy = False
        self.run_btn.disabled = False
        if self.progress.value is None:
            self.progress.value = 0
        if not self.status.value.startswith("녹취록만"):
            self.status.value = "실패"
        self.page.update()
        self._snack(msg, error=True)

    def _open_result(self, _e) -> None:
        if self.result_path:
            _open_path(self.result_path)

    async def _copy_result(self, _e) -> None:
        if self.result_path:
            await ft.Clipboard().set(self.result_path.read_text(encoding="utf-8"))
            self._snack("회의록을 클립보드에 복사했습니다.")


def main() -> None:
    ft.run(MeetingApp)
