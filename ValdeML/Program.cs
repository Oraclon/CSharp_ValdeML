using System;

namespace ValdeML
{
    class Program
    {
        static void Main(string[] args)
        {
            string path = @"C:\Users\Raphael\Documents\Datasets\testbinaryclassific.csv";
            StreamReader reader = new StreamReader(path);
            string[] lines = reader.ReadToEnd().Split("\n");
            
            MMODEL[] dataset = new MMODEL[lines.Length-1];
            for(int i = 0; i< lines.Length-1; i++)
            {
                if (!lines[i].Equals(""))
                {
                    MMODEL model = new MMODEL();
                    string[] str_lines = lines[i].Split(",");
                    double[] line = str_lines.Select(x => Convert.ToDouble(x)).ToArray();
                    model.input = line.Skip(1).ToArray();
                    model.target = line[0];
                    dataset[i] = model;
                }
            }

            dataset = new ZSCORE().Get(dataset);
            MMODEL[][] batches = new Batches().Get(dataset, 64);

            Grad grad = new Grad();
            grad.a = .4;
            BCM bcm = new BCM();
            bcm.Train(grad, batches);
        }
    }
}