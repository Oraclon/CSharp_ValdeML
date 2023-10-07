using System;

namespace ValdeML
{
    class Program
    {
        static void Main(string[] args)
        {
            Transposer transposer = new Transposer();
            Random random = new Random();
            int s = 5000;
            MMODEL[] dataset = new MMODEL[s];
            for(int i = 0; i< s; i++)
            {
                int x = i + 1;
                MMODEL model = new MMODEL();
                model.input = new double[] { x * 100, x * 10 };
                model.target = x * 2;
                dataset[i] = model;
            }

            
            Grad grad = new Grad();
            ZSCORE scaled = new ZSCORE();
            dataset = scaled.Get(dataset);
            dataset = dataset.OrderBy(_ => random.Next()).ToArray();
            grad.scalers = scaled.scalers;
            MMODEL[][] batches = new Batches().Get(dataset, 512);
            
            LRM lrm = new LRM();
            grad.a = 0.2;
            grad.UpdateW(dataset[0].input);
            lrm.Train(grad, batches);
            int testint = 1000000;
            double[] testinp = grad.ScaleInput(new double[] { testint * 100, testint * 10 });
            double prediction= lrm.Predict(grad, testinp);
        }
    }
}