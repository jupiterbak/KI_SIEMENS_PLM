:: The following sript works only under Windows
@echo off
SETLOCAL DisableDelayedExpansion
echo Setting global variables...
::variables
set SRC_DIR=./protobuf-definitions
set DST_DIR_C=./communicatorapi_cpp
set DST_DIR_P=./communicatorapi_python
set PROTO_PATH=./protobuf-definitions
set PYTHON_PACKAGE=rpc_communicator

set COMPILER="C:/Program Files/grpc\PlugIn"
echo Cleaning up target directories...
::clean
del /F /Q "%DST_DIR_C%"
del /F /Q "%DST_DIR_P%"
IF not exist %DST_DIR_C% (mkdir "%DST_DIR_C%")
IF not exist %DST_DIR_P% (mkdir "%DST_DIR_P%")

echo Generate proto objects in python and Ansi C...
::generate proto objects in python and Ansi C
for %%f in (%SRC_DIR%/*.proto) do (
  protoc --proto_path=%PROTO_PATH% --cpp_out=%DST_DIR_C% %SRC_DIR%/%%f
  protoc --proto_path=%PROTO_PATH% --python_out=%DST_DIR_P% %SRC_DIR%/%%f
)


echo Generate GRPC Services for python and Ansi C...
::grpc
set GRPC=FAPSPLMServives.proto
protoc --proto_path=%PROTO_PATH% --cpp_out=%DST_DIR_C% --grpc_out=%DST_DIR_C% %SRC_DIR%/%GRPC% --plugin=protoc-gen-grpc=%COMPILER%/grpc_cpp_plugin.exe
python -m grpc_tools.protoc --proto_path=%PROTO_PATH% --python_out=%DST_DIR_P%\ --grpc_python_out=%DST_DIR_P%\ %SRC_DIR%/%GRPC%


echo Generate the init file for the python module...
::Generate the init file for the python module

set init_file=%DST_DIR_P%\__init__.py
echo Deleting %init_file% ...
del /F "%init_file%"

for %%f in (%DST_DIR_P%/*.py) do (
    if "%%f"=="__init__.py" (
        echo Ignoring %%f ...
    )else (
        echo from .%%f import * >> "%DST_DIR_P%/__init__.py"
    )
)

ENDLOCAL