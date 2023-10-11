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
            DatasetMultFeatures data = new DatasetMultFeatures();
            data.Build(200000, 128, 2, "mean", true);
            MMODEL[][] batches  = data.batches;
            MMODEL[][] to_train = batches.Skip(0).Take(batches.Length - 10).ToArray();
            MMODEL[][] to_eval  = batches.Skip(batches.Length - 10).ToArray();
            Model model         = new Model();
            BCD bcm             = new BCD();

            model.SetLearningRate(.4);
            bcm.Train(model, to_train);
        }
    }
}
