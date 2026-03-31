# fix_load.py — รันครั้งเดียวเพื่อ re-save โมเดล
import keras
import numpy as np

# Patch BatchNormalization ให้กิน renorm args แล้วทิ้งทิ้ง
_orig_bn_init = keras.layers.BatchNormalization.__init__

def _patched_bn_init(self, **kwargs):
    kwargs.pop("renorm", None)
    kwargs.pop("renorm_clipping", None)
    kwargs.pop("renorm_momentum", None)
    _orig_bn_init(self, **kwargs)

keras.layers.BatchNormalization.__init__ = _patched_bn_init

# โหลดด้วย safe_mode=False
model = keras.saving.load_model(
    "fruit_cnn_model.keras",
    custom_objects={"BatchNormalization": keras.layers.BatchNormalization},
    safe_mode=False,
)

print("✅ โหลดสำเร็จ:", model.input_shape)

# Re-save เป็นไฟล์ใหม่ที่ compatible
model.save("fruit_cnn_model_fixed.keras")
print("✅ Save fixed model สำเร็จ")