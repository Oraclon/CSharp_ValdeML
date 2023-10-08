using System;

namespace ValdeML
{
    enum Fruit
    {
        orange,
        grapefruit
    }
    class Program
    {
        static void Main(string[] args)
        {
            string path = "/Users/angularnodedeveloper/Documents/datasets/citrus.csv";
            StreamReader reader = new StreamReader(path);
            string[] lines = reader.ReadToEnd().Split("\n").Skip(1).ToArray();

            MMODEL[] dataset = new MMODEL[lines.Length-1];
            for (int i = 0; i < lines.Length-1; i++)
            {
                if (!lines[i].Equals(""))
                {
                    MMODEL model = new MMODEL();
                    string[] str_lines = lines[i].Split(",");
                    model.input = str_lines.Skip(1).Select(x => Convert.ToDouble(x)).ToArray();
                    model.target= (int)(Fruit)Enum.Parse(typeof(Fruit), str_lines[0]);
                    dataset[i] = model;
                }
            }

            dataset = new ZSCORE().Get(dataset);
            MMODEL[][] batches = new Batches().Get(dataset, 128);
            Grad grad = new Grad();
            grad.a = .4;
            BCM bcm = new BCM();
            bcm.Train(grad, batches);
        }
    }
}
