import numpy as np
import torch
import math
import torch.nn.functional as F
from torch import nn

import warnings

warnings.filterwarnings('ignore')


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def uv2xyz(uv):
    xyz = np.zeros((*uv.shape[:-1], 3), dtype=np.float32)
    xyz[..., 0] = np.multiply(np.cos(uv[..., 1]), np.sin(uv[..., 0]))
    xyz[..., 1] = np.multiply(np.cos(uv[..., 1]), np.cos(uv[..., 0]))
    xyz[..., 2] = np.sin(uv[..., 1])
    return xyz


class Equi2Pers:
    def __init__(self, erp_size, fov, nrows, patch_size, shift=False, geom=None):
        assert geom in [None, 'cubemap', 'icosahedron'], 'Check geometry!'

        self.patch_size = pair(patch_size)
        erp_h, erp_w = erp_size
        height, width = pair(patch_size)
        fov_h, fov_w = pair(fov)
        FOV = torch.tensor([fov_w / 360.0, fov_h / 180.0], dtype=torch.float32)

        PI = math.pi
        PI_2 = math.pi * 0.5
        yy, xx = torch.meshgrid(torch.linspace(0, 1, height), torch.linspace(0, 1, width))
        screen_points = torch.stack([xx.flatten(), yy.flatten()], -1)

        use_icosahedron = geom == 'icosahedron'
        use_cubemap = geom == 'cubemap'
        assert not (shift and use_icosahedron), "Shifted tangent images can't be used with icosahedron geometry."
        assert not (shift and use_cubemap), "Shifted tangent images can't be used with icosahedron geometry."
        use_icosahedron and print("Using Icosahedron Geometry.")
        use_cubemap and print("Using Cubemap Geometry.")

        if use_cubemap:
            fov_h, fov_w = pair(115)  # 115 is the lower bound for FoV
            FOV = torch.tensor([fov_w / 360.0, fov_h / 180.0], dtype=torch.float32)

            num_rows = 3
            num_cols = [1, 4, 1]
            phi_centers = [-90, 0, 90]
        else:  # Tangent Images
            if nrows == 3:
                num_rows = 3
                num_cols = [3, 4, 3]
                phi_centers = [-60, 0, 60]
            elif nrows == 4:
                num_rows = 4
                if use_icosahedron:
                    num_cols = [5, 5, 5, 5]
                    fov_h, fov_w = pair(90)  # 90 is the lower bound for FoV
                    FOV = torch.tensor([fov_w / 360.0, fov_h / 180.0], dtype=torch.float32)
                else:
                    num_cols = [3, 6, 6, 3]
                phi_centers = [-67.5, -22.5, 22.5, 67.5]
            elif nrows == 5:
                num_rows = 5
                num_cols = [3, 6, 8, 6, 3]
                phi_centers = [-72.2, -36.1, 0, 36.1, 72.2]
            elif nrows == 6:
                num_rows = 6
                num_cols = [3, 8, 12, 12, 8, 3]
                phi_centers = [-75.2, -45.93, -15.72, 15.72, 45.93, 75.2]
            else:
                raise NotImplementedError("nrows", nrows, "is not valid.")
        phi_interval = 180 // num_rows
        all_combos = []
        erp_mask = []
        for i, n_cols in enumerate(num_cols):
            for j in np.arange(n_cols):
                theta_interval = 360 / n_cols
                if use_cubemap:
                    theta_add = theta_interval if n_cols == 4 else theta_interval / 2
                    theta_center = j * theta_interval + theta_add
                else:
                    if nrows == 4 and use_icosahedron:
                        if i == 1:  # Upper half of the equator
                            theta_center = j * theta_interval + theta_interval
                        elif i == 2:  # Lower half of the equator
                            theta_center = j * theta_interval + theta_interval / 2
                        else:  # poles, treat same as before
                            theta_center = j * theta_interval + theta_interval / 2
                    else:
                        if shift:
                            theta_center = j * theta_interval + theta_interval
                        else:
                            theta_center = j * theta_interval + theta_interval / 2

                center = [theta_center, phi_centers[i]]
                all_combos.append(center)
                up = phi_centers[i] + phi_interval / 2
                down = phi_centers[i] - phi_interval / 2
                left = theta_center - theta_interval / 2
                right = theta_center + theta_interval / 2
                up = int((up + 90) / 180 * erp_h)
                down = int((down + 90) / 180 * erp_h)
                left = int(left / 360 * erp_w)
                right = int(right / 360 * erp_w)
                mask = np.zeros((erp_h, erp_w), dtype=int)
                mask[down:up, left:right] = 1
                erp_mask.append(mask)
        all_combos = np.vstack(all_combos)
        num_patch = all_combos.shape[0]
        self.num_patch = num_patch

        center_point = torch.from_numpy(all_combos).float()  # -180 to 180, -90 to 90
        center_point[:, 0] = (center_point[:, 0]) / 360  # 0 to 1
        center_point[:, 1] = (center_point[:, 1] + 90) / 180  # 0 to 1

        cp = center_point * 2 - 1
        center_p = cp.clone()
        cp[:, 0] = cp[:, 0] * PI
        cp[:, 1] = cp[:, 1] * PI_2
        cp = cp.unsqueeze(1)
        convertedCoord = screen_points * 2 - 1
        convertedCoord[:, 0] = convertedCoord[:, 0] * PI
        convertedCoord[:, 1] = convertedCoord[:, 1] * PI_2
        convertedCoord = convertedCoord * (torch.ones(screen_points.shape, dtype=torch.float32) * FOV)
        convertedCoord = convertedCoord.unsqueeze(0).repeat(cp.shape[0], 1, 1)

        x = convertedCoord[:, :, 0]
        y = convertedCoord[:, :, 1]

        rou = torch.sqrt(x ** 2 + y ** 2)
        c = torch.atan(rou)
        sin_c = torch.sin(c)
        cos_c = torch.cos(c)
        lat = torch.asin(cos_c * torch.sin(cp[:, :, 1]) + (y * sin_c * torch.cos(cp[:, :, 1])) / rou)
        lon = cp[:, :, 0] + torch.atan2(x * sin_c,
                                        rou * torch.cos(cp[:, :, 1]) * cos_c - y * torch.sin(cp[:, :, 1]) * sin_c)
        lat_new = lat / PI_2
        lon_new = lon / PI
        lon_new[lon_new > 1] -= 2
        lon_new[lon_new < -1] += 2

        lon_new = lon_new.view(1, num_patch, height, width).permute(0, 2, 1, 3).contiguous().view(height,
                                                                                                  num_patch * width)
        lat_new = lat_new.view(1, num_patch, height, width).permute(0, 2, 1, 3).contiguous().view(height,
                                                                                                  num_patch * width)
        grid = torch.stack([lon_new, lat_new], -1)

        grid_tmp = torch.stack([lon, lat], -1)
        xyz = uv2xyz(grid_tmp)
        xyz = xyz.reshape((num_patch, height, width, 3)).transpose(0, 3, 1, 2)
        xyz = torch.Tensor(xyz)

        uv = grid.reshape(height, width, num_patch, 2).permute(2, 3, 0, 1)
        uv = uv.contiguous()

        self.grid = grid
        self.xyz = xyz
        self.uv = uv
        self.center_p = center_p
        #
        # print("Equi->Pers instance created with:\n"
        #       f"\tFoV: {fov}\n"
        #       f"\tn_patches: {num_patch}\n"
        #       f"\tpatch_size: {patch_size}\n-------")

    def project(self, erp_img):
        bs, _, erp_h, erp_w = erp_img.shape
        height, width = self.patch_size
        # grid = self.grid.unsqueeze(0).repeat(bs, 1, 1, 1).to(erp_img.device)
        grid = self.grid.unsqueeze(0).repeat(bs, 1, 1, 1).cuda()
        pers = F.grid_sample(erp_img, grid, mode='bilinear', padding_mode='border', align_corners=True)
        pers = F.unfold(pers, kernel_size=(height, width), stride=(height, width))
        pers = pers.reshape(bs, -1, height, width, self.num_patch)
        return pers

    def project_clip(self, erp_clip):
        bs, T, ch, erp_h, erp_w = erp_clip.shape
        erp_batched = erp_clip.reshape(bs * T, ch, erp_h, erp_w)
        height, width = self.patch_size
        # grid = self.grid.unsqueeze(0).repeat(bs * T, 1, 1, 1).to(erp_clip.device)
        grid = self.grid.unsqueeze(0).repeat(bs * T, 1, 1, 1).cuda()
        pers = F.grid_sample(erp_batched, grid, mode='bilinear', padding_mode='border', align_corners=True)
        pers = F.unfold(pers, kernel_size=(height, width), stride=(height, width))
        pers = pers.reshape(bs, T, -1, height, width, self.num_patch)
        return pers

    def get_spherical_embeddings(self):
        xy = self.xyz[:, :2, ...]
        xyz_h, xyz_w = self.patch_size
        center_points = self.center_p.reshape(-1, 2, 1, 1).repeat(1, 1, xyz_h, xyz_w)
        rho = torch.ones((self.num_patch, 1, xyz_h, xyz_w), dtype=torch.float32)
        new_xyz = torch.cat([xy, rho, center_points], 1)
        return new_xyz


class Pers2Equi(nn.Module):
    def __init__(self, erp_size, fov, nrows, patch_size, shift=False, geom=None):
        assert geom in [None, 'cubemap', 'icosahedron'], 'Check geometry!'
        super().__init__()
        height, width = pair(patch_size)
        fov_h, fov_w = pair(fov)
        self.erp_h, self.erp_w = pair(erp_size)
        FOV = torch.tensor([fov_w / 360.0, fov_h / 180.0], dtype=torch.float32)

        PI = math.pi
        PI_2 = math.pi * 0.5

        use_icosahedron = geom == 'icosahedron'
        use_cubemap = geom == 'cubemap'
        assert not (shift and use_icosahedron), "Shifted tangent images can't be used with icosahedron geometry."
        assert not (shift and use_cubemap), "Shifted tangent images can't be used with icosahedron geometry."
        use_icosahedron and print("Using Icosahedron Geometry.")
        use_cubemap and print("Using Cubemap Geometry.")

        if use_cubemap:
            num_rows = 3
            num_cols = [1, 4, 1]
            phi_centers = [-90, 0, 90]

            fov_h, fov_w = pair(115)  # 115 is the lower bound for FoV
            FOV = torch.tensor([fov_w / 360.0, fov_h / 180.0], dtype=torch.float32)
        else:  # Tangent Images
            if nrows == 3:
                num_cols = [3, 4, 3]
                phi_centers = [-59.6, 0, 59.6]
            elif nrows == 4:
                num_rows = 4
                if use_icosahedron:
                    num_cols = [5, 5, 5, 5]
                    fov_h, fov_w = pair(90)  # 90 is the lower bound for FoV
                    FOV = torch.tensor([fov_w / 360.0, fov_h / 180.0], dtype=torch.float32)

                else:
                    num_cols = [3, 6, 6, 3]
                phi_centers = [-67.5, -22.5, 22.5, 67.5]

            elif nrows == 5:
                num_cols = [3, 6, 8, 6, 3]
                phi_centers = [-72.2, -36.1, 0, 36.1, 72.2]
            elif nrows == 6:
                num_cols = [3, 8, 12, 12, 8, 3]
                phi_centers = [-75.2, -45.93, -15.72, 15.72, 45.93, 75.2]
            else:
                raise NotImplementedError("nrows", nrows, "is not valid")
        all_combos = []

        for i, n_cols in enumerate(num_cols):
            for j in np.arange(n_cols):
                theta_interval = 360 / n_cols
                if use_cubemap:
                    theta_add = theta_interval if n_cols == 4 else theta_interval / 2
                    theta_center = j * theta_interval + theta_add
                else:
                    if nrows == 4 and use_icosahedron:
                        if i == 1:  # Upper half of the equator
                            theta_center = j * theta_interval + theta_interval
                        elif i == 2:  # Lower half of the equator
                            theta_center = j * theta_interval + theta_interval / 2
                        else:  # poles, treat same as before
                            theta_center = j * theta_interval + theta_interval / 2
                    else:
                        if shift:
                            theta_center = j * theta_interval + theta_interval
                        else:
                            theta_center = j * theta_interval + theta_interval / 2

                center = [theta_center, phi_centers[i]]
                all_combos.append(center)

        all_combos = np.vstack(all_combos)
        n_patch = all_combos.shape[0]

        center_point = torch.from_numpy(all_combos).float()  # -180 to 180, -90 to 90
        center_point[:, 0] = (center_point[:, 0]) / 360  # 0 to 1
        center_point[:, 1] = (center_point[:, 1] + 90) / 180  # 0 to 1

        cp = center_point * 2 - 1
        cp[:, 0] = cp[:, 0] * PI
        cp[:, 1] = cp[:, 1] * PI_2
        cp = cp.unsqueeze(1)

        lat_grid, lon_grid = torch.meshgrid(torch.linspace(-PI_2, PI_2, self.erp_h),
                                            torch.linspace(-PI, PI, self.erp_w))
        lon_grid = lon_grid.float().reshape(1, -1)  # .repeat(num_rows*num_cols, 1)
        lat_grid = lat_grid.float().reshape(1, -1)  # .repeat(num_rows*num_cols, 1)
        cos_c = torch.sin(cp[..., 1]) * torch.sin(lat_grid) + torch.cos(cp[..., 1]) * torch.cos(
            lat_grid) * torch.cos(
            lon_grid - cp[..., 0])
        new_x = (torch.cos(lat_grid) * torch.sin(lon_grid - cp[..., 0])) / cos_c
        new_y = (torch.cos(cp[..., 1]) * torch.sin(lat_grid) - torch.sin(cp[..., 1]) * torch.cos(
            lat_grid) * torch.cos(
            lon_grid - cp[..., 0])) / cos_c
        new_x = new_x / FOV[0] / PI  # -1 to 1
        new_y = new_y / FOV[1] / PI_2
        cos_c_mask = cos_c.reshape(n_patch, self.erp_h, self.erp_w)
        cos_c_mask = torch.where(cos_c_mask > 0, 1, 0)

        w_list = torch.zeros((n_patch, self.erp_h, self.erp_w, 4), dtype=torch.float32)

        new_x_patch = (new_x + 1) * 0.5 * height
        new_y_patch = (new_y + 1) * 0.5 * width
        new_x_patch = new_x_patch.reshape(n_patch, self.erp_h, self.erp_w)
        new_y_patch = new_y_patch.reshape(n_patch, self.erp_h, self.erp_w)
        mask = torch.where((new_x_patch < width) & (new_x_patch > 0) & (new_y_patch < height) & (new_y_patch > 0),
                           1, 0)
        mask *= cos_c_mask

        x0 = torch.floor(new_x_patch).type(torch.int64)
        x1 = x0 + 1
        y0 = torch.floor(new_y_patch).type(torch.int64)
        y1 = y0 + 1

        x0 = torch.clamp(x0, 0, width - 1)
        x1 = torch.clamp(x1, 0, width - 1)
        y0 = torch.clamp(y0, 0, height - 1)
        y1 = torch.clamp(y1, 0, height - 1)

        wa = (x1.type(torch.float32) - new_x_patch) * (y1.type(torch.float32) - new_y_patch)
        wb = (x1.type(torch.float32) - new_x_patch) * (new_y_patch - y0.type(torch.float32))
        wc = (new_x_patch - x0.type(torch.float32)) * (y1.type(torch.float32) - new_y_patch)
        wd = (new_x_patch - x0.type(torch.float32)) * (new_y_patch - y0.type(torch.float32))

        wa = wa * mask.expand_as(wa)
        wb = wb * mask.expand_as(wb)
        wc = wc * mask.expand_as(wc)
        wd = wd * mask.expand_as(wd)

        w_list[..., 0] = wa
        w_list[..., 1] = wb
        w_list[..., 2] = wc
        w_list[..., 3] = wd

        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.w_list = w_list  # .to(device)
        self.mask = mask  # .to(device)

        # Overlapping region mask.
        pers_img = torch.ones(1, 1, height, width, n_patch, dtype=torch.float32)
        n_patch = pers_img.shape[-1]
        mask = self.mask.to(pers_img.device)

        z = torch.arange(n_patch)
        z = z.reshape(n_patch, 1, 1)
        Ia = pers_img[:, :, self.y0, self.x0, z]

        overlap_mask = Ia * mask.expand_as(Ia)
        overlap_mask = overlap_mask.permute(0, 1, 3, 4, 2)


        # normalize
        overlap_mask = overlap_mask.sum(-1)
        overlap_mask = (overlap_mask - overlap_mask.min()) / (overlap_mask.max() - overlap_mask.min())
        overlap_mask = torch.exp(overlap_mask)
        self.overlap_mask = overlap_mask[0]
        # overlapping region mask end.

    def project(self, pers_img):
        n_patch = pers_img.shape[-1]
        # w_list = self.w_list.to(pers_img.device)
        # mask = self.mask.to(pers_img.device)
        w_list = self.w_list.cuda()
        mask = self.mask.cuda()

        z = torch.arange(n_patch)
        z = z.reshape(n_patch, 1, 1)
        Ia = pers_img[:, :, self.y0, self.x0, z]
        Ib = pers_img[:, :, self.y1, self.x0, z]
        Ic = pers_img[:, :, self.y0, self.x1, z]
        Id = pers_img[:, :, self.y1, self.x1, z]

        output_a = Ia * mask.expand_as(Ia)
        output_b = Ib * mask.expand_as(Ib)
        output_c = Ic * mask.expand_as(Ic)
        output_d = Id * mask.expand_as(Id)

        output_a = output_a.permute(0, 1, 3, 4, 2)
        output_b = output_b.permute(0, 1, 3, 4, 2)
        output_c = output_c.permute(0, 1, 3, 4, 2)
        output_d = output_d.permute(0, 1, 3, 4, 2)
        w_list = w_list.permute(1, 2, 0, 3)
        w_list = w_list.flatten(2)
        w_list *= torch.gt(w_list, 1e-5).type(torch.float32)
        w_list = F.normalize(w_list, p=1, dim=-1).reshape(self.erp_h, self.erp_w, n_patch, 4)
        w_list = w_list.unsqueeze(0).unsqueeze(0)
        output = output_a * w_list[..., 0] + output_b * w_list[..., 1] + output_c * w_list[..., 2] + output_d * w_list[
            ..., 3]

        return output.sum(-1)

    def forward(self, x):
        return self.project(x)
