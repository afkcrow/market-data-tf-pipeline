"""
Shared test fixtures.

pytest-asyncio (>=0.23) manages the event loop on its own — defining a custom
event_loop fixture here is deprecated. Add a session-wide loop scope below if
that becomes necessary in the future.
"""
