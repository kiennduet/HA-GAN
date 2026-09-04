# HA-GAN cho volume BẤT ĐỐI XỨNG (anisotropic) 160 x 192 x 224
#
# ============================================================================
# FILE MỚI - fork từ models/Model_HA_GAN_128.py
# ============================================================================
# Bản gốc của HA-GAN chỉ hỗ trợ volume lập phương (128^3 hoặc 256^3). File này
# sửa lại cho volume 160x192x224 (không gian MNI, khớp với data đã tiền xử lý
# cho 3D StyleGAN).
#
# Vì sao sửa được sạch sẽ:
#   160/5 = 192/6 = 224/7 = 32 = 2^5
#   -> hệ số upsample ĐỒNG NHẤT trên cả 3 trục (5 lần nhân đôi), dù số voxel
#      mỗi trục khác nhau. Nhờ vậy toàn bộ chuỗi F.interpolate(scale_factor=2)
#      của bản gốc giữ nguyên, chỉ cần đổi shape khởi tạo 4x4x4 -> 5x6x7 và
#      tính lại kernel của các lớp conv "thu về 1 voxel" ở cuối Discriminator/
#      Encoder.
#
# Bảng shape tham chiếu (dùng để debug khi sửa tiếp):
#   base (sau fc1)          : (5,   6,   7)     = IMG / 32
#   h_latent / low-res full : (40,  48,  56)    = IMG / 4
#   full volume             : (160, 192, 224)
#   crop high-res (D^H, E^H): (20,  192, 224)   = 1/8 trục đầu, 2 trục kia full
#   crop tại h_latent       : (5,   48,  56)    = 1/8 trục đầu
#
# Quy ước trục crop: GIỮ NGUYÊN bản gốc - luôn crop theo trục KHÔNG GIAN ĐẦU
# TIÊN (dim=2 của tensor 5D [N,C,D0,D1,D2]), ở đây là trục 160. Hai trục còn
# lại (192, 224) luôn được tính ở full resolution.
# ============================================================================

import numpy as np
import torch
import os
from torch import nn
from torch import optim
from torch.nn import functional as F
from models.layers import SNConv3d, SNLinear

# === THAY ĐỔI: gom toàn bộ hằng số shape về một chỗ để dễ bảo trì ===========
IMG_SHAPE  = (160, 192, 224)                                # volume đầy đủ
BASE_SHAPE = (5, 6, 7)                                      # shape sau fc1 (IMG/32)
LOW_SHAPE  = tuple(d // 4 for d in IMG_SHAPE)               # (40,48,56) - h_latent & low-res full
CROP_FULL  = IMG_SHAPE[0] // 8                              # 20  - độ dày lát cắt ở full res
CROP_LOW   = LOW_SHAPE[0] // 8                              # 5   - độ dày lát cắt ở h_latent
LOW_RATIO  = IMG_SHAPE[0] // LOW_SHAPE[0]                   # 4   - tỉ lệ full/h_latent
CROP_IDX_MAX = IMG_SHAPE[0] * 7 // 8                        # 140 - dùng chuẩn hoá crop_idx
# ===========================================================================


class Code_Discriminator(nn.Module):
    # KHÔNG ĐỔI so với bản gốc (chỉ làm việc trên vector latent, không liên quan shape ảnh)
    def __init__(self, code_size, num_units=256):
        super(Code_Discriminator, self).__init__()

        self.l1 = nn.Sequential(SNLinear(code_size, num_units),
                                nn.LeakyReLU(0.2,inplace=True))
        self.l2 = nn.Sequential(SNLinear(num_units, num_units),
                                nn.LeakyReLU(0.2,inplace=True))
        self.l3 = SNLinear(num_units, 1)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)

        return x


class Sub_Encoder(nn.Module):
    """E^G: gộp 8 feature map của Encoder -> vector latent.

    THAY ĐỔI: input (40,48,56) thay vì (32,32,32); conv5 dùng kernel (5,6,7)
    thay vì kernel 4 vì hai trục cuối không còn bằng nhau.
    """
    def __init__(self, channel=256, latent_dim=1024):
        super(Sub_Encoder, self).__init__()

        self.relu = nn.ReLU()
        # in: [40,48,56] (= 8 crop của Encoder nối theo dim=2)
        self.conv2 = nn.Conv3d(channel//4, channel//4, kernel_size=4, stride=2, padding=1) # out:[20,24,28]
        self.bn2 = nn.GroupNorm(8, channel//4)
        self.conv3 = nn.Conv3d(channel//4, channel//2, kernel_size=4, stride=2, padding=1) # out:[10,12,14]
        self.bn3 = nn.GroupNorm(8, channel//2)
        self.conv4 = nn.Conv3d(channel//2, channel, kernel_size=4, stride=2, padding=1)    # out:[5,6,7]
        self.bn4 = nn.GroupNorm(8, channel)
        # THAY ĐỔI: kernel_size=4 (cubic) -> BASE_SHAPE=(5,6,7) để thu về 1 voxel
        self.conv5 = nn.Conv3d(channel, latent_dim, kernel_size=BASE_SHAPE, stride=1, padding=0) # out:[1,1,1]

    def forward(self, h):
        h = self.conv2(h)
        h = self.relu(self.bn2(h))
        h = self.conv3(h)
        h = self.relu(self.bn3(h))
        h = self.conv4(h)
        h = self.relu(self.bn4(h))
        # THAY ĐỔI: .squeeze() -> .flatten(1). Bản gốc dùng .squeeze() sẽ nuốt luôn
        # chiều batch khi batch_size=1 (gây lỗi lúc test trên CPU). Với batch>1 hai
        # cách cho kết quả y hệt nhau.
        h = self.conv5(h).flatten(1)
        return h


class Encoder(nn.Module):
    """E^H: mã hoá một lát cắt high-res (20,192,224) -> feature (5,48,56).

    THAY ĐỔI: chỉ đổi shape ở comment - kiến trúc giữ nguyên vì toàn bộ là conv
    stride 2 đối xứng, tự động chia đôi mọi trục.
    """
    def __init__(self, channel=64):
        super(Encoder, self).__init__()

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv3d(1, channel//2, kernel_size=4, stride=2, padding=1)         # in:[20,192,224], out:[10,96,112]
        self.bn1 = nn.GroupNorm(8, channel//2)
        self.conv2 = nn.Conv3d(channel//2, channel//2, kernel_size=3, stride=1, padding=1) # out:[10,96,112]
        self.bn2 = nn.GroupNorm(8, channel//2)
        self.conv3 = nn.Conv3d(channel//2, channel, kernel_size=4, stride=2, padding=1)    # out:[5,48,56]
        self.bn3 = nn.GroupNorm(8, channel)

    def forward(self, h):
        h = self.conv1(h)
        h = self.relu(self.bn1(h))

        h = self.conv2(h)
        h = self.relu(self.bn2(h))

        h = self.conv3(h)
        h = self.relu(self.bn3(h))
        return h


class Sub_Discriminator(nn.Module):
    """D^L: phân biệt thật/giả trên low-res full volume (40,48,56).

    THAY ĐỔI: conv5 dùng kernel (5,6,7) thay vì 4.
    """
    def __init__(self, num_class=0, channel=256):
        super(Sub_Discriminator, self).__init__()
        self.channel = channel
        self.num_class = num_class

        self.conv2 = SNConv3d(1, channel//4, kernel_size=4, stride=2, padding=1)           # in:[40,48,56], out:[20,24,28]
        self.conv3 = SNConv3d(channel//4, channel//2, kernel_size=4, stride=2, padding=1)  # out:[10,12,14]
        self.conv4 = SNConv3d(channel//2, channel, kernel_size=4, stride=2, padding=1)     # out:[5,6,7]
        # THAY ĐỔI: kernel_size=4 -> BASE_SHAPE=(5,6,7)
        self.conv5 = SNConv3d(channel, 1+num_class, kernel_size=BASE_SHAPE, stride=1, padding=0) # out:[1,1,1]

    def forward(self, h):
        h = F.leaky_relu(self.conv2(h), negative_slope=0.2)
        h = F.leaky_relu(self.conv3(h), negative_slope=0.2)
        h = F.leaky_relu(self.conv4(h), negative_slope=0.2)
        if self.num_class == 0:
            h = self.conv5(h).view((-1,1))
            return h
        else:
            h = self.conv5(h).view((-1,1+self.num_class))
            return h[:,:1], h[:,1:]


class Discriminator(nn.Module):
    """D^H (lát cắt high-res) + D^L (low-res full volume).

    THAY ĐỔI LỚN NHẤT của cả file nằm ở đây. Bản gốc giả định 2 trục không bị
    crop luôn BẰNG NHAU (128x128 hoặc 256x256) nên dùng chung một kernel scalar.
    Ở đây 192 != 224 nên phải tính lại từng lớp bằng out = (in + 2p - k)/s + 1.

    Chuỗi shape (trục crop, trục 192, trục 224):
        input  (20, 192, 224)
        conv2  (10,  96, 112)
        conv3  ( 5,  48,  56)
        conv4  ( 1,  24,  28)   <- kernel 5 trên trục crop để gộp hẳn về 1
        conv5  ( 1,  12,  14)
        conv6  ( 1,   6,   7)
        conv7  ( 1,   1,   1)   <- kernel (1,6,7)
    """
    def __init__(self, num_class=0, channel=512):
        super(Discriminator, self).__init__()
        self.channel = channel
        self.num_class = num_class

        # D^H
        self.conv2 = SNConv3d(1, channel//16, kernel_size=4, stride=2, padding=1)          # in:[20,192,224], out:[10,96,112]
        self.conv3 = SNConv3d(channel//16, channel//8, kernel_size=4, stride=2, padding=1) # out:[5,48,56]
        # THAY ĐỔI: kernel trục crop = CROP_LOW (5) với stride 1/padding 0 để thu
        # trục crop về đúng 1. Bản gốc dùng (2,4,4) hai lần vì trục crop của nó là
        # luỹ thừa của 2 (16 -> 8 -> 4 -> 2 -> 1); trục crop của ta là 20 -> 10 -> 5
        # (số lẻ) nên gộp một phát bằng kernel 5.
        self.conv4 = SNConv3d(channel//8, channel//4, kernel_size=(CROP_LOW,4,4), stride=(1,2,2), padding=(0,1,1)) # out:[1,24,28]
        self.conv5 = SNConv3d(channel//4, channel//2, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1))        # out:[1,12,14]
        self.conv6 = SNConv3d(channel//2, channel, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1))           # out:[1,6,7]
        # THAY ĐỔI: kernel (1,4,4) -> (1, 6, 7) = (1,) + BASE_SHAPE[1:]
        self.conv7 = SNConv3d(channel, channel//4, kernel_size=(1,)+BASE_SHAPE[1:], stride=1, padding=0)           # out:[1,1,1]
        self.fc1 = SNLinear(channel//4+1, channel//8)
        self.fc2 = SNLinear(channel//8, 1)
        if num_class>0:
            self.fc2_class = SNLinear(channel//8, num_class)

        # D^L
        self.sub_D = Sub_Discriminator(num_class)

    def forward(self, h, h_small, crop_idx):
        h = F.leaky_relu(self.conv2(h), negative_slope=0.2)
        h = F.leaky_relu(self.conv3(h), negative_slope=0.2)
        h = F.leaky_relu(self.conv4(h), negative_slope=0.2)
        h = F.leaky_relu(self.conv5(h), negative_slope=0.2)
        h = F.leaky_relu(self.conv6(h), negative_slope=0.2)
        # THAY ĐỔI: .squeeze() -> .flatten(1) (xem giải thích ở Sub_Encoder)
        h = F.leaky_relu(self.conv7(h), negative_slope=0.2).flatten(1)
        # THAY ĐỔI: (1) hằng số chuẩn hoá 112. -> CROP_IDX_MAX (=140 = 160*7/8);
        #           (2) bỏ .cuda() hardcode, dùng device/dtype của chính h để có
        #               thể chạy/test trên CPU.
        crop_feat = torch.full((h.size(0), 1), crop_idx / float(CROP_IDX_MAX),
                               device=h.device, dtype=h.dtype)
        h = torch.cat([h, crop_feat], 1)
        h = F.leaky_relu(self.fc1(h), negative_slope=0.2)
        h_logit = self.fc2(h)
        if self.num_class>0:
            h_class_logit = self.fc2_class(h)

            h_small_logit, h_small_class_logit = self.sub_D(h_small)
            return (h_logit+ h_small_logit)/2., (h_class_logit+ h_small_class_logit)/2.
        else:
            h_small_logit = self.sub_D(h_small)
            return (h_logit+ h_small_logit)/2.


class Sub_Generator(nn.Module):
    # KHÔNG ĐỔI so với bản gốc: toàn bộ là conv 3x3 stride 1, không phụ thuộc shape.
    # Nhận h_latent (40,48,56) -> low-res full volume 1 kênh cùng shape.
    def __init__(self, channel:int=16):
        super(Sub_Generator, self).__init__()
        _c = channel

        self.relu = nn.ReLU()
        self.tp_conv1 = nn.Conv3d(_c*4, _c*2, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.GroupNorm(8, _c*2)

        self.tp_conv2 = nn.Conv3d(_c*2, _c, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.GroupNorm(8, _c)

        self.tp_conv3 = nn.Conv3d(_c, 1, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, h):

        h = self.tp_conv1(h)
        h = self.relu(self.bn1(h))

        h = self.tp_conv2(h)
        h = self.relu(self.bn2(h))

        h = self.tp_conv3(h)
        h = torch.tanh(h)
        return h


class Generator(nn.Module):
    """G^A + G^H + G^L.

    THAY ĐỔI: fc1 sinh ra khối 5x6x7 thay vì 4x4x4; số lần nhân đôi giữ nguyên
    (3 lần trước khi tách nhánh, 2 lần sau khi crop) -> tổng 2^5 = 32.

    Chuỗi shape:
        fc1/view (  5,   6,   7)  ch=_c*16
        tp_conv1 (  5,   6,   7)
        up->conv2( 10,  12,  14)
        up->conv3( 20,  24,  28)
        up->conv4( 40,  48,  56)
        tp_conv5 ( 40,  48,  56)  = h_latent, ch=_c*2
          |- sub_G  -> h_small (40,48,56), 1 kênh          [nhánh low-res toàn cục]
          |- crop   -> ( 5,  48,  56)                      [nhánh high-res cục bộ]
        up->conv6( 10,  96, 112)
        up->conv7( 20, 192, 224)
    """
    def __init__(self, mode="train", latent_dim=1024, channel=32, num_class=0):
        super(Generator, self).__init__()
        _c = channel

        self.mode = mode
        self.relu = nn.ReLU()
        self.num_class = num_class

        # G^A and G^H
        # THAY ĐỔI: 4*4*4 -> 5*6*7 (BASE_SHAPE)
        self.fc1 = nn.Linear(latent_dim+num_class,
                             BASE_SHAPE[0]*BASE_SHAPE[1]*BASE_SHAPE[2]*_c*16)

        self.tp_conv1 = nn.Conv3d(_c*16, _c*16, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.GroupNorm(8, _c*16)

        self.tp_conv2 = nn.Conv3d(_c*16, _c*16, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.GroupNorm(8, _c*16)

        self.tp_conv3 = nn.Conv3d(_c*16, _c*8, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn3 = nn.GroupNorm(8, _c*8)

        self.tp_conv4 = nn.Conv3d(_c*8, _c*4, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn4 = nn.GroupNorm(8, _c*4)

        self.tp_conv5 = nn.Conv3d(_c*4, _c*2, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn5 = nn.GroupNorm(8, _c*2)

        self.tp_conv6 = nn.Conv3d(_c*2, _c, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn6 = nn.GroupNorm(8, _c)

        self.tp_conv7 = nn.Conv3d(_c, 1, kernel_size=3, stride=1, padding=1, bias=True)

        # G^L
        self.sub_G = Sub_Generator(channel=_c//2)

        # THAY ĐỔI: lưu lại để dùng trong .view() thay vì hardcode 512
        self._base_ch = _c*16

    def forward(self, h, crop_idx=None, class_label=None):

        # Generate from random noise
        if crop_idx != None or self.mode=='eval':
            if self.num_class > 0:
                h = torch.cat((h, class_label), dim=1)

            h = self.fc1(h)

            # THAY ĐỔI: view(-1,512,4,4,4) -> shape 5x6x7, và dùng self._base_ch
            # thay vì hardcode 512 (để đổi `channel` không làm vỡ code)
            h = h.view(-1, self._base_ch, *BASE_SHAPE)
            h = self.tp_conv1(h)
            h = self.relu(self.bn1(h))

            h = F.interpolate(h,scale_factor = 2)
            h = self.tp_conv2(h)
            h = self.relu(self.bn2(h))

            h = F.interpolate(h,scale_factor = 2)
            h = self.tp_conv3(h)
            h = self.relu(self.bn3(h))

            h = F.interpolate(h,scale_factor = 2)
            h = self.tp_conv4(h)
            h = self.relu(self.bn4(h))

            h = self.tp_conv5(h)
            h_latent = self.relu(self.bn5(h)) # (40,48,56), channel: _c*2

            if self.mode == "train":
                h_small = self.sub_G(h_latent)
                # THAY ĐỔI: crop_idx//4 -> crop_idx//LOW_RATIO, +4 -> +CROP_LOW
                # LƯU Ý (kế thừa từ bản gốc): crop_idx không nhất thiết chia hết cho
                # LOW_RATIO nên lát cắt sinh ra có thể lệch <= 3 voxel so với lát cắt
                # thật. Giữ nguyên hành vi gốc để dễ đối chiếu; nếu muốn khớp tuyệt
                # đối thì sample crop_idx theo bội của LOW_RATIO trong train.py.
                h = h_latent[:,:,crop_idx//LOW_RATIO:crop_idx//LOW_RATIO+CROP_LOW,:,:] # out: (5,48,56)
            else:
                h = h_latent

        # Generate from latent feature
        h = F.interpolate(h,scale_factor = 2)
        h = self.tp_conv6(h)
        h = self.relu(self.bn6(h)) # (10,96,112) khi train / (80,96,112) khi eval

        h = F.interpolate(h,scale_factor = 2)
        h = self.tp_conv7(h)

        h = torch.tanh(h) # (20,192,224) khi train / (160,192,224) khi eval

        if crop_idx != None and self.mode == "train":
            return h, h_small
        return h
