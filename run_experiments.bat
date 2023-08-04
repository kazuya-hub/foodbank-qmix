
@echo off

@REM for %%s in (10a10f_c6 10a10f_c7 10a10f_c8) do (
@REM     python .\src\main.py --algo qmix --env foodbank --situ "%%s" --wandb

@REM     rem バッチファイルが存在するディレクトリに中断用のファイル（stop）を作成する
@REM     if exist stop (
@REM         del stop
@REM         exit /b
@REM     )
@REM )



@REM for /l %%i in (1,1,7) do (

@REM     for %%s in (10a10f_c2 10a10f_c3 10a10f_c4 10a10f_c5 10a10f_c6 10a10f_c7 10a10f_c8) do (
@REM         python .\src\main.py --algo qmix --env foodbank --situ "%%s" --wandb

@REM         rem バッチファイルが存在するディレクトリに中断用のファイル（stop）を作成する
@REM         if exist stop (
@REM             del stop
@REM             exit /b
@REM         )
@REM     )
@REM )



for /l %%i in (1,1,7) do (

    python .\src\main.py --algo qmix --env foodbank --situ "10a10f_c8" --wandb

    rem バッチファイルが存在するディレクトリに中断用のファイル（stop）を作成する
    if exist stop (
        del stop
        exit /b
    )
)
