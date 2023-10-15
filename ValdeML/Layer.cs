using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace ValdeML
{
    public class Layer
    {
        #region Layer Constructor
        public Layer(int layer_size, int layer_id, Activation layer_node_activation, Optimizer node_optimizer)
        {
            Nodes = new Node[layer_size];
            TotalNodes = layer_size;

            LayerId = $"Layer: [{layer_id}]";
            LayerDescription = new LayerInfo(layer_node_activation);

            for (int i = 0; i < layer_size; i++)
            {
                Node node = new Node(i + 1, layer_node_activation, node_optimizer);
                Nodes[i] = node;
            }
        }
        #endregion
        #region Layer Variables
        public LayerInfo LayerDescription       { get; set; }
        Node[] Nodes                            { get; set; }
        public int TotalNodes                   { get; set; }
        public double[][] LayerDerivatives      { get; set; }
        public double[][] LayerActivations      { get; set; }
        public double[][] NodeDerivs            { get; set; }
        public string LayerId                   { get; set; }
        #endregion
        #region Layer Voids
        public void TrainNodes(double[][] inputs)
        {
            int TotalNodes = Nodes.Length;
            int TotalInputs = inputs.Length;

            //LayerActivations = new double[TotalNodes][];
            double[][] CollectedActivations = new double[TotalNodes][];
            LayerDerivatives = new double[TotalNodes][];

            for (int i = 0; i< TotalInputs; i++)
            {
                for (int j = 0; j < TotalNodes; j++)
                {
                    Node SelectedNode = Nodes[j];
                    SelectedNode.NodePredict(inputs[i], i, TotalInputs);
                }
            }
            for (int j = 0; j < TotalNodes; j++)
            {
                Node SelectedNode = Nodes[j];
                LayerDerivatives[j] = SelectedNode.Activations;
                CollectedActivations[j] = SelectedNode.ActivationDerivs;
            }

            LayerActivations = Transposer.TransposeList(CollectedActivations);
        }
        public void GetNodesDerivs(double[][] previous_layer_derivs, double[][] respect_to)
        {
            NodeDerivs = Nodes.Length.Equals(1) ? new double[1][] : new double[Nodes.Length][];
            
            for (int i = 0; i < TotalNodes; i++)
            {
                Node SelectedNode = Nodes[i];
                
                if (Nodes.Length.Equals(1))
                {
                    SelectedNode.NodeCalDerivs(previous_layer_derivs[0], respect_to);
                    NodeDerivs[0] = SelectedNode.NodeJ;
                }
                else
                {
                    double[] selected_node_deriv= new double[previous_layer_derivs[0].Length];
                    for (int j = 0; j < previous_layer_derivs.Length; j++)
                    {
                        SelectedNode.NodeCalDerivs(previous_layer_derivs[j], respect_to);
                        double[] SelectedNodeJ = SelectedNode.NodeJ;
                        selected_node_deriv = SelectedNodeJ;
                    }
                    NodeDerivs[i] = selected_node_deriv;
                }
            }
        }
        public void UpdateNodes(Model model)
        {
            for (int i = 0; i < Nodes.Length; i++)
            {
                Node SelectedNode = Nodes[i];
                SelectedNode.UpdateGradient(model);
            }
        }
        #endregion
    }
}
