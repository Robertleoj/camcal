uv pip install --force-reinstall --no-build-isolation -e .
pybind11-stubgen lensboy.lensboy_bindings -o src
# pybind11-stubgen bug: doesn't strip the module prefix from types nested inside
# container types (e.g. std::optional<T> -> T | None). Strip it manually.
if [[ "$OSTYPE" == darwin* ]]; then
    sed -i '' 's/lensboy\.lensboy_bindings\.//g' src/lensboy/lensboy_bindings.pyi
else
    sed -i 's/lensboy\.lensboy_bindings\.//g' src/lensboy/lensboy_bindings.pyi
fi