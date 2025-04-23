# -*- coding: utf-8 -*-
import numpy as np
import cv2 as cv
import random
import math

def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def NCC(img1, img2):
    return abs(np.mean(np.multiply((img1 - np.mean(img1)), (img2 - np.mean(img2)))) / (np.std(img1) * np.std(img2)))

def watermark_image(img, wm, key=50, bs=8, indx=0, indy=0, fact=8, b_cut=50):
    h_img, w_img = img.shape
    wm = cv.resize(wm, (64, 64))
    _, wm = cv.threshold(wm, 127, 255, cv.THRESH_BINARY)
    h_wm, w_wm = wm.shape

    h_core = h_img - b_cut * 2
    w_core = w_img - b_cut * 2
    blocks = (h_core // bs) * (w_core // bs)

    if (h_core * w_core // (bs * bs)) < (h_wm * w_wm):
        raise ValueError("Watermark too large for image.")

    random.seed(key)
    used_blocks = set()
    imf = img.astype(np.float32)
    total_blocks_needed = h_wm * w_wm

    for i in range(total_blocks_needed):
        wm_bit = 1 if wm[i // w_wm][i % w_wm] > 0 else 0

        while True:
            block_idx = random.randint(0, blocks - 1)
            if block_idx not in used_blocks:
                used_blocks.add(block_idx)
                break

        n_blocks = w_core // bs
        y = (block_idx // n_blocks) * bs + b_cut
        x = (block_idx % n_blocks) * bs + b_cut

        block = imf[y:y + bs, x:x + bs]
        dct_block = cv.dct(block / 1.0)
        coeff = dct_block[indx][indy] / fact

        if wm_bit:
            coeff = math.floor(coeff) | 1
        else:
            coeff = math.floor(coeff) & ~1

        dct_block[indx][indy] = coeff * fact
        block_idct = cv.idct(dct_block)
        imf[y:y + bs, x:x + bs] = block_idct

    imf = np.clip(imf, 0, 255).astype(np.uint8)
    return imf

def extract_watermark(img, size=(64, 64), key=50, bs=8, indx=0, indy=0, fact=8, b_cut=50):
    h_img, w_img = img.shape
    h_core = h_img - b_cut * 2
    w_core = w_img - b_cut * 2
    blocks = (h_core // bs) * (w_core // bs)

    h_wm, w_wm = size
    wm = np.zeros((h_wm, w_wm), dtype=np.uint8)

    random.seed(key)
    used_blocks = set()
    total_blocks_needed = h_wm * w_wm

    for i in range(total_blocks_needed):
        while True:
            block_idx = random.randint(0, blocks - 1)
            if block_idx not in used_blocks:
                used_blocks.add(block_idx)
                break

        n_blocks = w_core // bs
        y = (block_idx // n_blocks) * bs + b_cut
        x = (block_idx % n_blocks) * bs + b_cut

        block = img[y:y + bs, x:x + bs].astype(np.float32)
        dct_block = cv.dct(block / 1.0)
        coeff = dct_block[indx][indy] / fact
        wm_bit = 1 if int(coeff) % 2 == 1 else 0
        wm[i // w_wm][i % w_wm] = 255 if wm_bit else 0

    return wm

if __name__ == "__main__":
    cover = cv.imread("image2.jpg", cv.IMREAD_GRAYSCALE)
    watermark = cv.imread("watermark1.jpg", cv.IMREAD_GRAYSCALE)

    watermarked = watermark_image(cover, watermark)
    cv.imwrite("Watermarked_Image.jpg", watermarked)

    extracted = extract_watermark(watermarked)
    cv.imwrite("Extracted_Watermark.jpg", extracted)

    print("PSNR:", psnr(cover, watermarked))
    print("NCC:", NCC(cv.resize(watermark, (64, 64)), extracted))
