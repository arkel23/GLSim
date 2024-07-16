mkdir -p ckpts

wget https://huggingface.co/NYCU-PCSxNTHU-MIS/GLSim/resolve/main/cub_vit_b16_16_2.pth?download=true -O ckpts/cub_glsim_224.pth
wget https://huggingface.co/NYCU-PCSxNTHU-MIS/GLSim/resolve/main/dafb_vit_b16_16_2.pth?download=true -O ckpts/dafb_glsim.pth
wget https://huggingface.co/NYCU-PCSxNTHU-MIS/GLSim/resolve/main/inat17_vit_b16_16_2.pth?download=true -O ckpts/inat_glsim.pth
wget https://huggingface.co/NYCU-PCSxNTHU-MIS/GLSim/resolve/main/nabirds_vit_b16_16_2.pth?download=true -O ckpts/nabirds_glsim.pth
