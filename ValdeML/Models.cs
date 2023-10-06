using System;
namespace ValdeML
{
    class SCALER
    {
        internal double m { get; set; }
        internal double s { get; set; }
        internal double min { get; set; }
        internal double max { get; set; }
    }
    class SMODEL
    {
        internal double input { get; set; }
        internal double target { get; set; }
    }
    class MMODEL
    {
        internal double[] input { get; set; }
        internal double[] target { get; set; }
    }
}

