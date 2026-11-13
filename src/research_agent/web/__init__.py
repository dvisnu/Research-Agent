"""The local web UI (`agent ui`): start and resume runs, watch them stage by stage, and
browse what they found, down to the sentence each claim rests on.

The server (``app``) reads the same store the pipeline writes; runs it starts are
`agent run` / `agent resume` child processes (``jobs``). The React app it serves is
built from ``frontend/`` into ``web/static``.
"""
