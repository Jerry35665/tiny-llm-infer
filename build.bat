@echo off
rem one-click build for tiny-llm-infer (Ninja + MSVC via vcvars64)
rem usage: build.bat [extra cmake configure args]
chcp 65001 >nul
setlocal

set "VS_ROOT=C:\Program Files\Microsoft Visual Studio\18\Community"
set "CMAKE_EXE=%VS_ROOT%\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
set "NINJA_EXE=%VS_ROOT%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"

if not exist "%CMAKE_EXE%" (
  echo [ERROR] cmake.exe not found at:
  echo   %CMAKE_EXE%
  echo If your Visual Studio path differs, edit VS_ROOT in this script.
  exit /b 1
)
if not exist "%NINJA_EXE%" set "NINJA_EXE=ninja"

call "%VS_ROOT%\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 (
  echo [ERROR] vcvars64.bat failed
  exit /b 1
)

if not exist "build\CMakeCache.txt" (
  "%CMAKE_EXE%" -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release "-DCMAKE_MAKE_PROGRAM=%NINJA_EXE%" %*
  if errorlevel 1 exit /b 1
)

"%CMAKE_EXE%" --build build
if errorlevel 1 exit /b 1

echo.
echo Done. Run: build\tli.exe --info
