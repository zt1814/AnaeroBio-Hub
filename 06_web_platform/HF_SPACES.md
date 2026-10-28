# Deploying AnaeroBio-Hub to Hugging Face Spaces (Docker SDK)

1. Create a Space: **Docker** SDK, hardware *CPU basic*. Choose private or public in the Space settings; decide before submission whether reviewers need an open link.
2. Build the Space repo from the project (the repo root is the Docker build context):
   ```
   <space repo>/
     Dockerfile                      <- copy of 06_web_platform/Dockerfile.hf
     README.md                       <- front matter below
     06_web_platform/                (py files, ui/, .streamlit/, requirements.txt, multimodal_v4_bundle.pkl,
                                      docker_entrypoint_hf.sh, nginx.hf.conf)
     01_final_dataset/anaerobic_multimodal_benchmark.db
     03_priors_and_metabolism/QSAR_Observed_Microbial_Metabolism_Pathways.csv
     site/                           (landing page)
   ```
   `README.md` front matter:
   ```
   ---
   title: AnaeroBio-Hub
   emoji: 🧪
   colorFrom: green
   colorTo: blue
   sdk: docker
   app_port: 7860
   pinned: false
   ---
   ```
   Files over 10 MB (the .db and the .pkl) need Git LFS/Xet: `git lfs track "*.db" "*.pkl"` before the first commit.
3. Space → Settings → **Variables and secrets**:
   - `PUBLIC_BASE_URL` = `https://<user>-<space>.hf.space` (shown in the API examples)
   - `ENABLE_PUBLIC_SUBMISSION` = `false` (default)
   - `ANBIOGATE_ADMIN_TOKEN` = a **Secret**, only if you need curator approval on the Space (otherwise leave unset)
4. The Space filesystem is ephemeral: anything written to the SQLite DB (community submissions, retrained bundles)
   is lost on restart. That is why public submission stays closed on the Space.
5. Free Spaces sleep after 48 h without visits; the first request wakes them (~1 min).
