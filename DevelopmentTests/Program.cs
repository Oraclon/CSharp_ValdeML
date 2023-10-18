namespace DevelopmentTests
{
    enum Activation
    {
        Tahn,
        Sigmoid
    }

    class Model
    { }
    class LayerModel
    {
        public LayerModel(Activation activation, int total_nodes)
        {
            nodeActivation = activation;
            totalNodes = total_nodes;
        }

        public Activation nodeActivation { get; set; }
        public int totalNodes { get; set; }
    }

    class Program
    {
        public class Node
        {
            public Node(int attached_layer, int node_id, Activation activation)
            {
                nodeId = node_id;
                attachedLayer = attached_layer;
                activationId = (int)activation;
                activationName = activation.ToString();
            }
            public int nodeId { get; set; }
            public int attachedLayer { get; set; }
            public int activationId { get; set; }
            public string activationName { get; set; }
        }

        public class Layer
        {
            public Layer(int layer_id, LayerModel layerModel)
            {
                layerId = layer_id;
                totalNodes = layerModel.totalNodes;

                nodes = new Node[totalNodes];

                for (int i = 0; i < totalNodes; i++)
                {
                    Node node = new Node(layerId, i + 1, layerModel.nodeActivation);
                    nodes[i] = node;
                }
            }

            public int layerId { get; set; }
            public int totalNodes { get; set; }
            public Node[] nodes { get; set; }
        }

        public class MLCore
        {
            public List<LayerModel> layerModels = new List<LayerModel>();
            public Layer[] layers { get; set; }

            public void AddLayer(Activation activation, int totalNodes)
            {
                LayerModel layerModel = new LayerModel(activation, totalNodes);
                layerModels.Add(layerModel);
            }

            public void Build()
            {
                int totalLayers = layerModels.Count();
                layers = new Layer[totalLayers];
                for (int i = 0; i < totalLayers; i++)
                {
                    LayerModel lModel = layerModels[i];
                    Layer layer = new Layer(i + 1, lModel);
                    layers[i] = layer;
                }
            }
        }

        static void Main(string[] args)
        {

            MLCore mlCore = new MLCore();

            mlCore.AddLayer(Activation.Tahn, 4);
            mlCore.AddLayer(Activation.Tahn, 4);
            mlCore.AddLayer(Activation.Tahn, 4);
            mlCore.AddLayer(Activation.Tahn, 4);
            mlCore.AddLayer(Activation.Sigmoid, 1);

            mlCore.Build();
            //double[] nodesList = new double[] { 30,30,1 };
            //int totalLayers = nodesList.Length;

            //Layer[] layers = new Layer[totalLayers];

            //for (int i = 0; i < totalLayers; i++)
            //{
            //    Layer layer = new Layer(i + 1, (int)nodesList[i]);
            //    layers[i] = layer;
            //}


        }
    }
}