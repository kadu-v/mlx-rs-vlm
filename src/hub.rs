use std::path::PathBuf;

use anyhow::Result;
use hf_hub::HFClientSync;
use mlx_rs as mlx;

pub fn download_model(repo_id: &str, revision: Option<&str>) -> Result<PathBuf> {
    let client = HFClientSync::new()?;
    let (owner, name) = repo_id
        .split_once('/')
        .ok_or_else(|| anyhow::anyhow!("invalid repo_id: {}", repo_id))?;
    let repo = client.model(owner, name);
    let path = repo
        .snapshot_download()
        .maybe_revision(revision.map(String::from))
        .maybe_allow_patterns(Some(vec![
            "*.safetensors".into(),
            "*.json".into(),
            "*.jinja".into(),
        ]))
        .send()?;
    Ok(path)
}
