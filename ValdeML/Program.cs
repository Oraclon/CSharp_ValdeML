namespace ValdeML
{
    class Program
    {
        static void Main(string[] args)
        {
            Dataset dataSet = new Dataset();
            dataSet.BuildDemo(300000, 128, 2, Scaler.ZScore, true);
        }
    }
}
