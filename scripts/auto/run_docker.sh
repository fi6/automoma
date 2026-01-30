docker run -d \
    --name automoma_cron_job \
    --restart unless-stopped \
    -v /home/xinhai:/home/xinhai \
    -v /home/xinhai/.ssh:/root/.ssh \
    auto-worker \
    bash /home/xinhai/projects/automoma/scripts/auto/auto_pipeline.sh