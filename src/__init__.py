"""Top-level package for the concept-graphs application.

Implementation modules are organized in subpackages such as ``src.core``,
``src.common``, and ``src.storage``.  A lightweight import hook keeps older
pickle files loadable when they reference former ``src.<module>`` paths.
"""

import importlib
import importlib.abc
import importlib.util
import sys

_LEGACY_MODULE_TARGETS = {
    "src.cluster_functions": "src.core.cluster_functions",
    "src.data_functions": "src.core.data_functions",
    "src.embedding_functions": "src.core.embedding_functions",
    "src.graph_functions": "src.core.graph_functions",
    "src.integration_functions": "src.core.integration_functions",
    "src.marqo_external_utils": "src.storage.marqo_external_utils",
}


class _LegacyModuleRedirector(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Redirect imports of former top-level modules to their new packages."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in _LEGACY_MODULE_TARGETS:
            return None
        return importlib.util.spec_from_loader(fullname, self)

    def create_module(self, spec):
        target_module = importlib.import_module(_LEGACY_MODULE_TARGETS[spec.name])
        sys.modules[spec.name] = target_module
        return target_module

    def exec_module(self, module):
        return None


if not any(isinstance(finder, _LegacyModuleRedirector) for finder in sys.meta_path):
    sys.meta_path.insert(0, _LegacyModuleRedirector())
