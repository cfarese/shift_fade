## Streamlit bidirectional component: HTML table with clickable player names.
##
## Player names render as styled link buttons. Clicking one sends the player_id
## back through Streamlit's component protocol -- same WebSocket session, no
## page reload, session state fully preserved.

from __future__ import annotations

from pathlib import Path

import streamlit.components.v1 as components

_FRONTEND = Path(__file__).parent / "clickable_table_frontend"
_component = components.declare_component("clickable_table", path=str(_FRONTEND))


def clickable_player_table(
    rows: list[dict],
    headers: list[str],
    key: str | None = None,
) -> int | None:
    ## rows: list of {"id": player_id (int), "values": [col1_str, col2_str, ...]}
    ## headers: column header labels -- first header corresponds to the name column
    ## returns the player_id of the clicked name, or None if nothing was clicked
    return _component(rows=rows, headers=headers, key=key, default=None)
