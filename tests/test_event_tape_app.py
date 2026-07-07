"""Smoke test — validates EventTape wiring under the real app fixture."""

import pytest

from tests.test_app import app, patched_data  # noqa: F401  (fixtures)
from widgets.event_tape import EventTape
from widgets.command_bar import CommandSubmitted


@pytest.mark.asyncio
async def test_eventtape_smoke(app):
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        tape = app.query_one("#event-tape", EventTape)
        assert tape.display is True
        # Push events and confirm they render into the tape's own content.
        app.event_feed.add("mover", "MSFT -3.6% mover", "negative")
        app.event_feed.add("alert", "🔔 AAPL crossed above 200.00", "critical")
        tape._repaint()
        await pilot.pause()
        r1 = str(tape.render())
        assert "MSFT" in r1 or "AAPL" in r1

        # Toggle off / on via the typed command.
        app.post_message(CommandSubmitted("events", ["off"], raw="events off"))
        await pilot.pause()
        assert tape.display is False and app._event_tape_enabled is False
        app.post_message(CommandSubmitted("events", ["on"], raw="events on"))
        await pilot.pause()
        assert tape.display is True

        # Ctrl+E action toggles too.
        app.action_toggle_event_tape()
        assert app._event_tape_enabled is False
        app.action_toggle_event_tape()
        assert app._event_tape_enabled is True

        # Time-travel gate — frozen note, no live scroll.
        app._tt_state = object()
        tape._repaint()
        await pilot.pause()
        rtt = str(tape.render())
        assert "paused" in rtt.lower() or "⏸" in rtt
        app._tt_state = None
