namespace ValdeML
{
    class Program
    {
        enum Fruits
        {
            orange,
            grapefruit
        }
        static void Main(string[] args)
        {
            Random r = new Random();
            string path = @"/Users/angularnodedeveloper/Projects/datasets/datasets/citrus.csv";
            StreamReader fileData = new StreamReader(path);
            string[] dataLines = fileData.ReadToEnd().Split("\n").Skip(1).ToArray();

            MMODEL[] dataSet = new MMODEL[dataLines.Length - 1];
            for(int i = 0; i< dataSet.Length; i++)
            {
                string[] lineData = dataLines[i].Split(",");
                MMODEL mod = new MMODEL();
                mod.target = (int)(Fruits)Enum.Parse(typeof(Fruits), lineData[0]);
                mod.input = lineData.Skip(1).Select(x => Convert.ToDouble(x)).ToArray();
                dataSet[i] = mod;
            }

            ZSCORE scaler = new ZSCORE();
            dataSet = scaler.Get(dataSet).OrderBy(_=> r.Next()).ToArray();
            MMODEL[][] batches = Batches.Get(dataSet, 128);
            MMODEL[][] toTrain = batches.Take(batches.Length - 4).ToArray();
            MMODEL[][] toEval = batches.Skip(batches.Length - 4).ToArray();


            Layer layer1 = new Layer(4, Activation.Tanh);
            Layer layer2 = new Layer(4, Activation.Tanh);
            Layer layer3 = new Layer(1, Activation.Sigmoid);

            Model model = new Model(Errors.LogLoss);
            model.Learning = .4;

            while (model.Error >= 0)
            {
                model.Epoch++;
                for (model.BatchId = 0; model.BatchId < toTrain.Length; model.BatchId++)
                {
                    MMODEL[] batch = toTrain[model.BatchId];
                    double[][] inputs = batch.Select(x => x.input).ToArray();
                    double[] targets = batch.Select(x => x.target).ToArray();
                    model.BatchSize = batch.Length;

                    layer1.NodesPredict(inputs);
                    layer2.NodesPredict(layer1.nodeActivations);
                    layer3.NodesPredict(layer2.nodeActivations);

                    model.CalculateError(layer3.nodeActivations, targets);

                    layer3.NodesCalcDeltas(model.ErrorDerivs, layer2.nodeActivations);
                    layer2.NodesCalcDeltas(layer3.nodeDeltas, layer1.nodeActivations);
                    layer1.NodesCalcDeltas(layer2.nodeDeltas, inputs);

                    layer3.NodesUpdate(model);
                    layer2.NodesUpdate(model);
                    layer1.NodesUpdate(model);

                    if (model.Error <= Math.Pow(10, -3))
                        break;
                }
                if (model.Error <= Math.Pow(10, -3))
                {
                    EvalTraining(toEval);
                    break;
                }
                string res = $"{model.Epoch}, {model.BatchId}, {model.Error}";
                Console.Write("\r{0} ", res);
            }

            void EvalTraining(MMODEL[][] toEval)
            {
                int correct = 0;
                int wrong = 0;
                double acc = 0;
                //I know i shouldn't use this. But i needed a fast solution for that.
                foreach (MMODEL[] batch in toEval)
                {
                    foreach(MMODEL item in batch)
                    {
                        //Evaluation Code
                    }
                }
            }
        }
    }
}
