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
        public Layer(int total_nodes, Activation activation)
        {
            totalNodes = total_nodes;
            nodes      = new Node[total_nodes];

            for (int i = 0; i < totalNodes; i++)
            {
                Node node = new Node(i + 1, activation);
                nodes[i] = node;
            }


            layerNodeActivation = activation.ToString();
        }
        #endregion

        #region Layer Common Variables
        public int totalNodes             { get; set; }
        public string layerNodeActivation { get; set; }
        public double[] evalActivations   { get; set; }
        #endregion

        #region Layer Nodes Variables
        public Node[] nodes               { get; set; }
        public double[][] nodeActivations { get; set; }
        public double[][] nodeDeltas      { get; set; }
        #endregion

        #region Layer Voids
        public void NodesPredict(double[][] inputs)
        {
            double[][] node_activations = new double[totalNodes][];

            for (int i = 0; i < totalNodes; i++)
            {
                nodes[i].NodePredict(inputs);
                node_activations[i] = nodes[i].nodeActivations;
            }

            nodeActivations = Transposer.TransposeList(node_activations);
        }

        public void NodesCalcDeltas(double[][] previous_derivatives, double[][] respect_to)
        {
            double[][] node_deltas = new double[totalNodes][];
            for (int i = 0; i < totalNodes; i++)
            {
                nodes[i].NodeCalcDeltas(previous_derivatives, respect_to);
                node_deltas[i] = nodes[i].nodeDeltas;
            }
            nodeDeltas = node_deltas;
        }

        public void NodesUpdate(Model model)
        {
            for (int i = 0; i < totalNodes; i++)
            {
                nodes[i].NodeUpdate(model);
            }
        }

        public void NodesEvaluate(double[] input)
        {
            double[] nodeEvalActivations = new double[totalNodes];
            for (int i = 0; i < totalNodes; i++)
            {
                nodeEvalActivations[i] = nodes[i].NodeEval(input);
            }
            evalActivations = nodeEvalActivations;
        }
        #endregion
    }
}
