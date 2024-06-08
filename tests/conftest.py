"""Configuration of pytest."""

from pytest import DoctestItem


# To order test modules logically during execution
def pytest_collection_modifyitems(items):  # type: ignore
    """Modifies test items in place to ensure test modules run in a given order."""

    # Ensure test_delayed runs last to avoid TearDown errors:
    # https://github.com/GlacioHack/geoutils/issues/545

    # We get names for all module test items (all tests except doctests)
    module_mapping = {item: item.module.__name__ for item in items if not isinstance(item, DoctestItem)}

    # We isolate items for doctests, we'll add them back later
    nonmodule_items = [item for item in items if isinstance(item, DoctestItem)]

    # We put all test_delayed items at the end of the module item list
    module_names = list(module_mapping.values())
    module_items = list(module_mapping.keys())

    module_items_reordered = [it for k, it in enumerate(module_items) if module_names[k] != "test_delayed"] + [
        it for k, it in enumerate(module_items) if module_names[k] == "test_delayed"
    ]

    # And write back items in that order, with doctests first
    items[:] = nonmodule_items + module_items_reordered
