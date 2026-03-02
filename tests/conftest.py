def pytest_collection_modifyitems(items: list) -> None:
    """Reorder tests so integration tests run last."""
    regular = []
    integration = []
    for item in items:
        if "test_integration" in str(item.fspath):
            integration.append(item)
        else:
            regular.append(item)
    items[:] = regular + integration
