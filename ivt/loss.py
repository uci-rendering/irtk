import torch

def l1_loss(target_image, rendered_image):
    return (target_image - rendered_image).abs().mean() # NOTE

def mesh_laplacian_smoothing(verts, faces):
    # compute L once per mesh subdiv.
    with torch.no_grad():
        V, F = verts.shape[0], faces.shape[0]
        face_verts = verts[faces]
        v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]
        # Side lengths of each triangle, of shape (sum(F_n),)
        # A is the side opposite v1, B is opposite v2, and C is opposite v3
        A = (v1 - v2).norm(dim=1)
        B = (v0 - v2).norm(dim=1)
        C = (v0 - v1).norm(dim=1)
        # Area of each triangle (with Heron's formula); shape is (sum(F_n),)
        s = 0.5 * (A + B + C)
        # note that the area can be negative (close to 0) causing nans after sqrt()
        # we clip it to a small positive value
        area = (s * (s - A) * (s - B) * (s - C)).clamp_(min=1e-6).sqrt()
        # Compute cotangents of angles, of shape (sum(F_n), 3)
        A2, B2, C2 = A * A, B * B, C * C
        cota = (B2 + C2 - A2) / area
        cotb = (A2 + C2 - B2) / area
        cotc = (A2 + B2 - C2) / area
        cot = torch.stack([cota, cotb, cotc], dim=1)
        cot /= 4.0
        # Construct a sparse matrix by basically doing:
        # L[v1, v2] = cota
        # L[v2, v0] = cotb
        # L[v0, v1] = cotc
        ii = faces[:, [1, 2, 0]]
        jj = faces[:, [2, 0, 1]]
        idx = torch.stack([ii, jj], dim=0).view(2, F * 3)
        L = torch.sparse.FloatTensor(idx, cot.view(-1), (V, V))
        # Make it symmetric; this means we are also setting
        # L[v2, v1] = cota
        # L[v0, v2] = cotb
        # L[v1, v0] = cotc
        L += L.t()
        norm_w = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
        idx = norm_w > 1e-6
        norm_w[idx] = 1.0 / norm_w[idx]
    loss = L.mm(verts) * norm_w - verts
    loss = loss.norm(dim=1)
    return loss.mean()

def total_variation_loss(texture, texture_mask):
    m = (texture_mask[:-1, :-1] & texture_mask[1:, :-1] & texture_mask[:-1, 1:])
    loss = (texture[:-1, :-1] - texture[1:, :-1])[m].abs().mean() +\
           (texture[:-1, :-1] - texture[:-1, 1:])[m].abs().mean()
    return loss