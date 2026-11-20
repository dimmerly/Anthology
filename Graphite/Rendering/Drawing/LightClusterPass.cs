using System;

using Prowl.Graphite.ShaderDef;

namespace Prowl.Graphite.Rendering;

/// <summary>
/// Bins a view's local lights and decals into its clusters before its features draw, one dispatch a layer, so fragments
/// read short lists of what is near them rather than each walking the light and decal trees. See Graphite/LightClusters.glsl, whose layout
/// these constants match, and Graphite/Internal/LightClusters.glsl.
/// </summary>
internal sealed class LightClusterPass(InternalCompute program) : IDisposable
{
    public const uint ClustersX = 16, ClustersY = 9, ClustersZ = 24;
    public const uint ClusterCount = ClustersX * ClustersY * ClustersZ;
    public const uint LightSlots = 64, DecalSlots = 32;
    public const uint Slots = LightSlots + DecalSlots;
    public const uint Header = 4;

    /// <summary>The uvec2 slots of one layer's block: its header, then every cluster's slots.</summary>
    public const uint LayerSlots = Header + ClusterCount * Slots;

    /// <summary>The bytes of the clusters of a view of so many layers.</summary>
    public static uint Size(int layers) => (uint)Math.Max(layers, 1) * LayerSlots * 8;

    /// <summary>
    /// What a view without clusters binds: a header that puts every fragment in the first slice, and clusters there
    /// holding no lights or decals.
    /// </summary>
    public static uint[] Empty()
    {
        uint[] words = new uint[(Header + ClustersX * ClustersY * Slots) * 2];
        words[2] = BitConverter.SingleToUInt32Bits(1f);
        words[3] = BitConverter.SingleToUInt32Bits(1f);
        return words;
    }

    private static readonly PropertyID s_clusters = "Graphite_LightClusterWrite", s_receivers = "_ReceiverLayers";
    private readonly PropertySet _properties = new();

    /// <summary>Adds the pass building a view's clusters, before its first feature.</summary>
    public void Add(ViewGraph view)
    {
        CameraFrame camera = view.Camera;
        if (camera.LightClusters is not { } clusters)
            return;
        FramePipeline frame = view.Frame;
        FrameGraph graph = view.Graph;
        graph.AddPass("Graphite.LightClusters", builder =>
        {
            frame.Buffers.ReadWorld(graph, builder, camera.World);
            foreach (DeviceBuffer buffer in camera.Buffers.Items)
            {
                if (buffer != clusters)
                    builder.Read(graph.Import(buffer), BufferAccess.AllReads);
            }
            builder.Write(graph.Import(clusters), BufferAccess.ShaderRead | BufferAccess.ShaderWrite);
        }, context =>
        {
            CommandBuffer cmd = context.Command;
            view.BeginRecording(context);
            cmd.SetComputeShader(program.Program);
            _properties.SetBuffer(s_clusters, clusters);
            uint receivers = camera.World.World.Render.ReceiverLayers(camera.Desc.CullingMask) | camera.Sink.ItemLayers;
            _properties.SetInt(s_receivers, unchecked((int)receivers));
            for (uint layer = 0; layer < (uint)Math.Max(camera.LayerCount, 1); layer++)
            {
                camera.Drawer.BindTables(cmd, layer);
                cmd.SetProperties(_properties);
                cmd.Dispatch((ClusterCount + 63) / 64, 1, 1);
            }
        });
    }

    public void Dispose() => program.Dispose();
}
