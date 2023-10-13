using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ValdeML
{
    public class Layer
    {
        #region Layer Constructor
        public Layer(int layer_size, int layer_id, Activation layer_activation, Optimizer Optimizer)
        {

        }
        #endregion
        #region Layer Variables
        Node[] Nodes { get; set; }
        public double[][] Predictions { get; set; }
        public double[][] Activations { get; set; }
        public double[][] ActDerivs { get; set; }
        public double[][] NodeDerivs { get; set; }
        public double[][] NodeDerivsPow { get; set; }
        public double[][] NodeDerivsW { get; set; }
        public string LayerId { get; set; }
        #endregion
    }
}
