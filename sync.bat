@echo off
rem sync.bat "commit message"  —— 每完成一步后运行：提交并推送到 GitHub
rem 用法: sync.bat feat(kernels): attention naive
chcp 65001 >nul
if "%~1"=="" (
  echo 用法: sync.bat "commit message"
  exit /b 1
)
git add -A
git diff --cached --quiet
if errorlevel 1 (
  git commit -m "%~1"
) else (
  echo 没有需要提交的改动
)
git push
if errorlevel 1 (
  echo 推送失败 —— 确认已运行过一次: gh auth login
  exit /b 1
)
echo 已推送到 GitHub。
