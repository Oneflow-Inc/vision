python3 -m pytest -p no:randomly -p no:cacheprovider --max-worker-restart=0 \
-x --capture=sys -v --benchmark-disable-gc --benchmark-warmup=on \
            --benchmark-min-rounds=200 --benchmark-max-time=3  $1