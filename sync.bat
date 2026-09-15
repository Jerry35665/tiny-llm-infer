@echo off
rem sync.bat "commit message"  -- stage all, commit, push to GitHub
rem usage: sync.bat feat(kernels): attention naive
if "%~1"=="" (
  echo usage: sync.bat "commit message"
  exit /b 1
)
git add -A
git diff --cached --quiet
if errorlevel 1 (
  git commit -m "%~1"
) else (
  echo nothing to commit
)
git push
if errorlevel 1 (
  echo push failed - check remote with: git remote -v
  exit /b 1
)
echo pushed to GitHub.
