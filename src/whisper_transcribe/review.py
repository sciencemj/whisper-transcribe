"""Review a transcript against its recording and fix it while listening.

Plays the original audio, highlights the segment being spoken and lets
a person correct each segment in place. Edits are saved back to
`<name>_segments.json`, from which the minutes can be rebuilt.
"""

import asyncio
import bisect
import time
from pathlib import Path
from typing import Callable, Optional

import flet as ft
import flet_audio as fa

from .segments import SUFFIX, EditableSegment, TranscriptDoc

SPEEDS = ["0.75", "1.0", "1.25", "1.5", "2.0"]
SKIP_SECONDS = 5.0
# Players report their position only about once a second on some
# platforms, so the position is interpolated between reports.
TICK_SECONDS = 0.15
# Only this many rows are rendered at once: a long meeting has well over a
# thousand segments, which makes the client sluggish, and scroll_to() can't
# reach rows a lazy ListView hasn't built yet.
WINDOW = 80
AUTOSAVE_SECONDS = 2.0

SHORTCUTS = "Esc 재생/일시정지 · Ctrl/⌘+J/L 5초 이동 · Enter 확인 후 다음 문장 · Ctrl/⌘+S 저장"


def _clock(seconds: float) -> str:
    s = int(seconds)
    h, m, sec = s // 3600, s % 3600 // 60, s % 60
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


def _ms(seconds: float) -> ft.Duration:
    return ft.Duration(milliseconds=int(seconds * 1000))


class SegmentRow:
    def __init__(self, view: "ReviewView", index: int, seg: EditableSegment):
        self.view = view
        self.index = index
        self.seg = seg
        self.time_btn = ft.TextButton(
            _clock(seg.start),
            tooltip="이 부분부터 재생",
            on_click=lambda _: view.page.run_task(view.play_from, index),
        )
        self.field = ft.TextField(
            value=seg.text,
            multiline=True,
            min_lines=1,
            dense=True,
            expand=True,
            border=ft.UnderlineInputBorder(),
            on_change=self._on_change,
            on_focus=self._on_focus,
            on_blur=self._on_blur,
        )
        self.reviewed = ft.Checkbox(value=seg.reviewed, tooltip="확인 완료", on_change=self._on_reviewed)
        self.restore = ft.IconButton(
            ft.Icons.UNDO,
            icon_size=18,
            tooltip="원래 인식 결과로 되돌리기",
            visible=seg.edited,
            on_click=self._on_restore,
        )
        self.container = ft.Container(
            key=ft.ScrollKey(index),
            padding=ft.Padding.symmetric(horizontal=8, vertical=2),
            border_radius=8,
            content=ft.Row(
                vertical_alignment=ft.CrossAxisAlignment.START,
                controls=[self.time_btn, self.field, self.restore, self.reviewed],
            ),
        )
        self._refresh_marks()

    def set_active(self, active: bool) -> None:
        self.container.bgcolor = ft.Colors.SECONDARY_CONTAINER if active else None
        if self.view.in_window(self.index):
            self.container.update()

    def _refresh_marks(self) -> None:
        self.restore.visible = self.seg.edited
        self.restore.tooltip = f"원래 인식 결과로 되돌리기: {self.seg.original}"
        self.time_btn.style = ft.ButtonStyle(color=ft.Colors.TERTIARY) if self.seg.edited else None

    def _on_change(self, e) -> None:
        value = e.control.value
        if "\n" in value:
            # Enter confirms the segment and moves on. (Flet's `shift_enter`
            # would do this, but it breaks focus()/on_focus on the field.)
            value = value.replace("\n", " ").strip()
            self.field.value = value
            self.field.update()
            self.view.page.run_task(self.view.review_and_next, self.index)
        self.seg.text = value
        self.view.mark_dirty()

    def _on_focus(self, _e) -> None:
        self.view.focused = self.index

    def _on_blur(self, _e) -> None:
        if self.view.focused == self.index:
            self.view.focused = None
        if self.seg.text != self.seg.text.strip():
            self.seg.text = self.field.value = self.seg.text.strip()
            self.view.mark_dirty()
        self._refresh_marks()
        self.container.update()
        self.view.save_if_dirty()

    def _on_reviewed(self, e) -> None:
        self.seg.reviewed = e.control.value
        self.view.mark_dirty()
        self.view.save_if_dirty()

    def _on_restore(self, _e) -> None:
        self.seg.text = self.field.value = self.seg.original
        self._refresh_marks()
        self.container.update()
        self.view.mark_dirty()
        self.view.save_if_dirty()


class ReviewView:
    """The "녹취록 검토" tab. `on_rebuild(path)` regenerates the minutes."""

    def __init__(self, page: ft.Page, on_rebuild: Callable[[Path], None]):
        self.page = page
        self.on_rebuild = on_rebuild
        self.doc: Optional[TranscriptDoc] = None
        self.doc_path: Optional[Path] = None
        self.rows: list[SegmentRow] = []
        self.starts: list[float] = []
        self.audio: Optional[fa.Audio] = None

        # Playback state; position is base_pos + elapsed * rate while playing.
        self.started = False
        self.playing = False
        self.duration = 0.0
        self.rate = 1.0
        self.base_pos = 0.0
        self.base_time = time.monotonic()
        self.dragging = False
        self.current = -1
        self.focused: Optional[int] = None
        self.dirty = False
        self.last_edit = 0.0

        self.picker = ft.FilePicker()
        self.file_label = ft.Text("열린 녹취록 없음", weight=ft.FontWeight.W_600, expand=True, no_wrap=True)
        self.stats = ft.Text("", color=ft.Colors.ON_SURFACE_VARIANT)
        self.save_state = ft.Text("", color=ft.Colors.ON_SURFACE_VARIANT, size=12)
        self.save_btn = ft.OutlinedButton("저장", icon=ft.Icons.SAVE, disabled=True, on_click=lambda _: self.save())
        self.rebuild_btn = ft.FilledTonalButton(
            "회의록 다시 만들기", icon=ft.Icons.AUTO_AWESOME, disabled=True, on_click=self._rebuild
        )

        self.play_btn = ft.IconButton(
            ft.Icons.PLAY_CIRCLE_FILLED, icon_size=40, tooltip="재생 / 일시정지 (Esc)", on_click=self._toggle_clicked
        )
        self.slider = ft.Slider(
            min=0,
            max=1,
            value=0,
            expand=True,
            on_change_start=self._drag_start,
            on_change=self._drag,
            on_change_end=self._drag_end,
        )
        self.time_text = ft.Text("00:00 / 00:00", font_family="monospace")
        self.speed = ft.Dropdown(
            label="속도",
            value="1.0",
            width=110,
            dense=True,
            options=[ft.DropdownOption(key=s, text=f"{s}x") for s in SPEEDS],
            on_select=self._on_speed,
        )
        self.follow = ft.Switch(label="재생 위치 따라가기", value=True)
        self.player = ft.Column(
            disabled=True,
            spacing=4,
            controls=[
                ft.Row(
                    [
                        ft.IconButton(ft.Icons.REPLAY_5, tooltip="5초 뒤로 (Ctrl/⌘+J)", on_click=lambda _: self.page.run_task(self.skip, -SKIP_SECONDS)),
                        self.play_btn,
                        ft.IconButton(ft.Icons.FORWARD_5, tooltip="5초 앞으로 (Ctrl/⌘+L)", on_click=lambda _: self.page.run_task(self.skip, SKIP_SECONDS)),
                        self.slider,
                        self.time_text,
                        self.speed,
                    ]
                ),
                ft.Row(
                    [self.follow, ft.Container(expand=True), ft.Text(SHORTCUTS, size=12, color=ft.Colors.ON_SURFACE_VARIANT)]
                ),
            ],
        )
        self.list = ft.Column(scroll=ft.ScrollMode.AUTO, expand=True, spacing=0)
        self.win_start = 0
        self.prev_btn = ft.TextButton(
            "이전 문장 더 보기", icon=ft.Icons.EXPAND_LESS, on_click=lambda _: self.page.run_task(self._shift_window, -1)
        )
        self.next_btn = ft.TextButton(
            "다음 문장 더 보기", icon=ft.Icons.EXPAND_MORE, on_click=lambda _: self.page.run_task(self._shift_window, 1)
        )
        self.empty = ft.Container(
            expand=True,
            alignment=ft.Alignment.CENTER,
            content=ft.Column(
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                tight=True,
                controls=[
                    ft.Icon(ft.Icons.HEADPHONES, size=48, color=ft.Colors.ON_SURFACE_VARIANT),
                    ft.Text("회의록을 생성하거나, 녹취록 파일(*_segments.json)을 열어 주세요."),
                    ft.Text("원본 녹음을 들으면서 문장마다 바로 고칠 수 있습니다.", color=ft.Colors.ON_SURFACE_VARIANT),
                ],
            ),
        )

        self.control = ft.Container(
            expand=True,
            padding=16,
            content=ft.Column(
                expand=True,
                controls=[
                    ft.Row(
                        [
                            ft.IconButton(ft.Icons.FOLDER_OPEN, tooltip="녹취록 열기", on_click=self._pick),
                            self.file_label,
                            self.stats,
                            self.save_state,
                            self.save_btn,
                            self.rebuild_btn,
                        ]
                    ),
                    ft.Card(content=ft.Container(padding=ft.Padding.symmetric(horizontal=12, vertical=8), content=self.player)),
                    ft.Card(
                        expand=True,
                        content=ft.Container(padding=8, content=ft.Stack([self.list, self.empty], expand=True)),
                    ),
                ],
            ),
        )
        page.run_task(self._ticker)

    # ---- loading / saving ------------------------------------------------
    async def load(self, doc_path: Path) -> None:
        self.save_if_dirty()
        await self._release_audio()
        try:
            doc = TranscriptDoc.load(doc_path)
        except (OSError, ValueError, KeyError, TypeError) as e:
            self._snack(f"녹취록을 열 수 없습니다: {e}", error=True)
            return

        self.doc, self.doc_path = doc, doc_path
        self.rows = [SegmentRow(self, i, seg) for i, seg in enumerate(doc.segments)]
        self.starts = [seg.start for seg in doc.segments]
        self._set_window(0)
        self.empty.visible = not self.rows
        self.current = -1
        self.started = self.playing = False
        self.base_pos = 0.0
        self.duration = doc.duration
        self.file_label.value = doc_path.name
        self.save_btn.disabled = self.rebuild_btn.disabled = False
        self.save_state.value = ""
        self._update_stats()

        audio_path = doc.audio_path(doc_path)
        if audio_path.is_file():
            self.audio = fa.Audio(
                src=str(audio_path),
                playback_rate=self.rate,
                on_position_change=self._on_position,
                on_state_change=self._on_state,
                on_duration_change=self._on_duration,
            )
            self.player.disabled = False
        else:
            self.player.disabled = True
            self._snack(f"원본 녹음을 찾을 수 없어 재생할 수 없습니다: {doc.audio}", error=True)
        self.page.update()
        self._show_position(0.0)

    async def _release_audio(self) -> None:
        if self.audio is not None:
            try:
                await self.audio.release()
            except Exception:  # the client may already have dropped it
                pass
            self.audio = None

    def mark_dirty(self) -> None:
        self.last_edit = time.monotonic()
        if not self.dirty:
            self.dirty = True
            self.save_state.value = "저장 안 됨"
            self.save_state.update()

    def save_if_dirty(self) -> None:
        if self.dirty:
            self.save()

    def save(self) -> None:
        if not (self.doc and self.doc_path):
            return
        try:
            self.doc.save(self.doc_path)
        except OSError as e:
            self._snack(f"저장 실패: {e}", error=True)
            return
        self.dirty = False
        self.save_state.value = f"저장됨 {time.strftime('%H:%M:%S')}"
        self._update_stats()
        self.page.update()

    def _update_stats(self) -> None:
        segs = self.doc.segments if self.doc else []
        reviewed = sum(s.reviewed for s in segs)
        edited = sum(s.edited for s in segs)
        self.stats.value = f"확인 {reviewed}/{len(segs)} · 수정 {edited}"

    async def _pick(self, _e) -> None:
        files = await self.picker.pick_files(
            dialog_title="녹취록 열기",
            file_type=ft.FilePickerFileType.CUSTOM,
            allowed_extensions=["json"],
        )
        if files:
            path = Path(files[0].path)
            if not path.name.endswith(SUFFIX):
                self._snack(f"'*{SUFFIX}' 파일을 선택하세요.", error=True)
                return
            await self.load(path)

    def _rebuild(self, _e) -> None:
        self.save_if_dirty()
        if self.doc_path:
            self.on_rebuild(self.doc_path)

    # ---- playback --------------------------------------------------------
    def position(self) -> float:
        pos = self.base_pos
        if self.playing:
            pos += (time.monotonic() - self.base_time) * self.rate
        return max(0.0, min(pos, self.duration)) if self.duration else max(pos, 0.0)

    def _set_base(self, pos: float) -> None:
        self.base_pos = pos
        self.base_time = time.monotonic()

    async def toggle(self) -> None:
        if not self.audio:
            return
        if self.playing:
            await self.audio.pause()
        elif self.started:
            await self.audio.resume()
        else:
            await self._play_at(self.position())

    async def _play_at(self, pos: float) -> None:
        self._set_base(pos)
        await self.audio.play(position=_ms(pos))
        self.started = True

    async def seek(self, pos: float) -> None:
        if not self.audio:
            return
        pos = max(0.0, min(pos, self.duration or pos))
        self._set_base(pos)
        if self.started:
            await self.audio.seek(_ms(pos))
        self._show_position(pos, scroll=True)

    async def skip(self, delta: float) -> None:
        await self.seek(self.position() + delta)

    async def play_from(self, index: int) -> None:
        if not self.audio:
            return
        pos = self.doc.segments[index].start
        if self.started:
            await self.seek(pos)
            if not self.playing:
                await self.audio.resume()
        else:
            await self._play_at(pos)
            self._show_position(pos)

    def _on_position(self, e) -> None:
        self._set_base(e.position / 1000)
        if not self.playing and not self.dragging:
            self._show_position(self.base_pos)

    def _on_state(self, e) -> None:
        pos = self.position()
        self.playing = e.state == fa.AudioState.PLAYING
        if e.state == fa.AudioState.COMPLETED:
            self.started = False
            pos = 0.0
        self._set_base(pos)
        self.play_btn.icon = ft.Icons.PAUSE_CIRCLE_FILLED if self.playing else ft.Icons.PLAY_CIRCLE_FILLED
        self.play_btn.update()

    def _on_duration(self, e) -> None:
        self.duration = e.duration.in_milliseconds / 1000
        self._show_position(self.position())

    def _on_speed(self, e) -> None:
        self._set_base(self.position())
        self.rate = float(e.control.value)
        if self.audio:
            self.audio.playback_rate = self.rate
            self.audio.update()

    def _toggle_clicked(self, _e) -> None:
        self.page.run_task(self.toggle)

    def _drag_start(self, _e) -> None:
        self.dragging = True

    def _drag(self, e) -> None:
        self.time_text.value = f"{_clock(e.control.value)} / {_clock(self.duration)}"
        self.time_text.update()

    async def _drag_end(self, e) -> None:
        self.dragging = False
        await self.seek(e.control.value)

    async def _ticker(self) -> None:
        while True:
            await asyncio.sleep(TICK_SECONDS)
            try:
                if self.playing and not self.dragging:
                    self._show_position(self.position(), scroll=True)
                # Autosave shortly after typing stops, so closing the window
                # mid-edit loses at most a couple of seconds of work.
                if self.dirty and time.monotonic() - self.last_edit > AUTOSAVE_SECONDS:
                    self.save()
            except Exception:  # keep ticking even if one UI update fails
                pass

    # ---- row window --------------------------------------------------------
    @property
    def win_end(self) -> int:
        return min(self.win_start + WINDOW, len(self.rows))

    def in_window(self, index: int) -> bool:
        return self.win_start <= index < self.win_end

    def _set_window(self, start: int) -> None:
        self.win_start = max(0, min(start, len(self.rows) - WINDOW))
        controls = [row.container for row in self.rows[self.win_start : self.win_end]]
        if self.win_start > 0:
            controls.insert(0, ft.Row([self.prev_btn], alignment=ft.MainAxisAlignment.CENTER))
        if self.win_end < len(self.rows):
            controls.append(ft.Row([self.next_btn], alignment=ft.MainAxisAlignment.CENTER))
        self.list.controls = controls

    async def _reveal(self, index: int, duration: int = 300) -> None:
        """Scroll row `index` into view, moving the rendered window if needed."""
        if not self.in_window(index):
            self._set_window(index - 10)
            self.list.update()
            await asyncio.sleep(0.05)  # let the client lay out the new rows
        await self.list.scroll_to(scroll_key=ft.ScrollKey(index), duration=duration)

    async def _shift_window(self, direction: int) -> None:
        if self.focused is not None:
            self.save_if_dirty()
        anchor = self.win_start if direction < 0 else self.win_end - 1
        self._set_window(self.win_start + direction * WINDOW // 2)
        self.list.update()
        await asyncio.sleep(0.05)
        await self.list.scroll_to(scroll_key=ft.ScrollKey(anchor), duration=0)

    def _show_position(self, pos: float, scroll: bool = False) -> None:
        self.slider.max = max(self.duration, 1)
        self.slider.value = min(pos, self.slider.max)
        self.time_text.value = f"{_clock(pos)} / {_clock(self.duration)}"
        self.slider.update()
        self.time_text.update()

        index = bisect.bisect_right(self.starts, pos) - 1
        if index == self.current or not self.rows:
            return
        if 0 <= self.current < len(self.rows):
            self.rows[self.current].set_active(False)
        self.current = index
        if index >= 0:
            self.rows[index].set_active(True)
            # Don't yank the list away from a segment the user is editing.
            if scroll and self.follow.value and self.focused is None:
                self.page.run_task(self._reveal, index)

    # ---- keyboard ----------------------------------------------------------
    async def handle_key(self, e: ft.KeyboardEvent) -> None:
        if not self.doc:
            return
        cmd = e.ctrl or e.meta
        if e.key == "Escape" or (cmd and e.key == "K"):
            await self.toggle()
        elif cmd and e.key == "J":
            await self.skip(-SKIP_SECONDS)
        elif cmd and e.key == "L":
            await self.skip(SKIP_SECONDS)
        elif cmd and e.key == "S":
            self.save()
        elif cmd and e.key == "Enter" and self.focused is None:
            await self.review_and_next()

    async def review_and_next(self, index: Optional[int] = None) -> None:
        """Mark a segment (default: the one playing) reviewed and focus the next."""
        if index is None:
            index = self.focused if self.focused is not None else self.current
        if not 0 <= index < len(self.rows):
            return
        row = self.rows[index]
        row.seg.reviewed = row.reviewed.value = True
        if self.in_window(index):
            row.reviewed.update()
        self.mark_dirty()
        self.save()
        if index + 1 < len(self.rows):
            await self._reveal(index + 1, duration=200)
            await self.rows[index + 1].field.focus()

    def _snack(self, message: str, error: bool = False) -> None:
        self.page.show_dialog(
            ft.SnackBar(
                ft.Text(message),
                bgcolor=ft.Colors.ERROR_CONTAINER if error else None,
                duration=ft.Duration(seconds=8 if error else 4),
            )
        )
