from huggingface_hub import login, hf_hub_download, HfApi
log_name = ""
# upload the model
api = HfApi()

# fileuploader
api.upload_file(
    path_or_fileobj=f"logs/{log_name}/version_0/checkpoints/epoch=0-step=12207.ckpt",
    path_in_repo=f"2nd_epoch/{log_name}.ckpt",
    repo_id="pt-sk/GPT2_pretrained_finewebedu10B",
    repo_type="model",
)