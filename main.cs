namespace timeHelpers
{
  public static class Helpers
  {
    public static Int16 ToDate(this DateTime dateTime)
    {
      DateTime baseLine = new DateTime(2025, 1,1, 0,0,0, datetime.Kind);
      int epoch = (dateTime - baseLine).Days;
      return epoch;
    }
  }
}
