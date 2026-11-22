// This file is part of the Prowl Game Engine
// Licensed under the MIT License. See the LICENSE file in the project root for details.

using System;
using System.Threading.Tasks;

using Prowl.Photonic.Rasterization;
using Prowl.Vector;

namespace Prowl.Photonic.Imaging;

/// <summary>
/// Smooths a page's indirect estimate with an edge-avoiding à-trous wavelet filter (Dammertz et al. 2010), guided by
/// each texel's world position and normal instead of its colour. Indirect light changes slowly across a surface, so
/// averaging over a wide footprint removes the speckle a few hundred paths leave, while the geometric weights keep
/// the average on one surface: texels on another plane, facing another way, or on an unrelated chart packed next to
/// this one in the atlas never mix in. Direct light is left alone, so shadow edges stay sharp.
/// </summary>
internal static class IndirectDenoiser
{
    // B3 spline, the usual à-trous kernel.
    private static readonly float[] Kernel = [1f / 16f, 1f / 4f, 3f / 8f, 1f / 4f, 1f / 16f];

    // A tap one step away covers about one texel of surface; UV stretch can make it more, so allow some slack.
    private const float StretchAllowance = 1.75f;

    /// <summary>
    /// Writes direct + filtered indirect into the page's working buffer for every covered texel. Each pass doubles the
    /// tap spacing, so <paramref name="passes"/> passes reach about 4 * (2^passes - 1) texels.
    /// </summary>
    public static void Apply(TargetWorkspace ws, int passes, ParallelOptions opts)
    {
        int W = ws.Width, H = ws.Height, n = W * H;
        Float3[] sums = ws.IndirectSum!;
        int[] counts = ws.IndirectSampleCount!;
        Float3[] direct = ws.DirectCache!;
        bool[] covered = ws.Covered;
        TexelSample[] samples = ws.Samples;
        Float3[] a = ws.DenoiseA ??= new Float3[n];
        Float3[] b = ws.DenoiseB ??= new Float3[n];

        Parallel.For(0, n, opts, i => a[i] = covered[i] && counts[i] > 0 ? sums[i] * (1f / counts[i]) : Float3.Zero);

        for (int pass = 0; pass < passes; pass++)
        {
            int step = 1 << pass;
            Float3[] src = a, dst = b;
            Parallel.For(0, H, opts, y =>
            {
                for (int x = 0; x < W; x++)
                {
                    int i = y * W + x;
                    if (!covered[i]) { dst[i] = Float3.Zero; continue; }

                    ref readonly TexelSample sp = ref samples[i];
                    float texel = MathF.Max(sp.WorldRadius * 2f, 1e-5f);
                    float planeTolerance = texel * 1.5f;

                    Float3 sum = Float3.Zero;
                    float weightSum = 0f;
                    for (int ky = -2; ky <= 2; ky++)
                    {
                        int qy = y + ky * step;
                        if ((uint)qy >= (uint)H) continue;
                        for (int kx = -2; kx <= 2; kx++)
                        {
                            int qx = x + kx * step;
                            if ((uint)qx >= (uint)W) continue;
                            int q = qy * W + qx;
                            if (!covered[q]) continue;

                            ref readonly TexelSample sq = ref samples[q];
                            float facing = Float3.Dot(sp.Normal, sq.Normal);
                            if (facing < 0.9f) continue;

                            // Far further away in the world than in the atlas: another chart packed alongside.
                            Float3 offset = sq.Position - sp.Position;
                            float reach = texel * step * StretchAllowance * MathF.Sqrt(kx * kx + ky * ky) + planeTolerance;
                            if (Float3.LengthSquared(offset) > reach * reach) continue;

                            // Off this texel's plane: a parallel surface, or round a corner the normals agree on.
                            if (MathF.Abs(Float3.Dot(sp.Normal, offset)) > planeTolerance) continue;

                            float w = Kernel[ky + 2] * Kernel[kx + 2] * facing * facing;
                            sum += src[q] * w;
                            weightSum += w;
                        }
                    }
                    dst[i] = weightSum > 0f ? sum * (1f / weightSum) : src[i];
                }
            });
            a = dst;
            b = src;
        }

        float[] pixels = ws.Working();
        Float3[] result = a;
        Parallel.For(0, n, opts, i =>
        {
            if (!covered[i]) return;
            Float3 v = direct[i] + result[i];
            pixels[i * 3] = v.X;
            pixels[i * 3 + 1] = v.Y;
            pixels[i * 3 + 2] = v.Z;
        });
    }
}
