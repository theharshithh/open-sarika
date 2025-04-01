unset CUDA_VISIBLE_DEVICES
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="0,1"
export TORCH_USE_CUDA_DSA=1
echo "CUDA_VISIBLE_DEVICES is set to: $CUDA_VISIBLE_DEVICES"

torchrun \
    --nproc_per_node=2 \
    --master_port=29500 \
    trainer.py
