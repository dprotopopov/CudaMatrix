using System;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Windows.Forms;

namespace CudaMatrix.Editor
{
    public partial class MatrixForm : Form
    {
        private static readonly Random Rnd = new Random();

        private readonly MatrixIO _dataGridViewMatrixA = new MatrixIO
        {
            ColumnHeadersHeightSizeMode = DataGridViewColumnHeadersHeightSizeMode.AutoSize,
            Dock = DockStyle.Fill,
            Name = "dataGridViewMatrix",
            RowTemplate = {Height = 20},
            TabIndex = 0
        };

        private readonly MatrixIO _dataGridViewMatrixB = new MatrixIO
        {
            ColumnHeadersHeightSizeMode = DataGridViewColumnHeadersHeightSizeMode.AutoSize,
            Dock = DockStyle.Fill,
            Name = "dataGridViewMatrix",
            RowTemplate = {Height = 20},
            TabIndex = 0
        };

        private readonly MatrixIO _dataGridViewMatrixC = new MatrixIO
        {
            ColumnHeadersHeightSizeMode = DataGridViewColumnHeadersHeightSizeMode.AutoSize,
            Dock = DockStyle.Fill,
            Name = "dataGridViewMatrix",
            RowTemplate = {Height = 20},
            TabIndex = 0
        };

        public MatrixForm()
        {
            InitializeComponent();
            tabPageA.Controls.Add(_dataGridViewMatrixA);
            tabPageB.Controls.Add(_dataGridViewMatrixB);
            tabPageC.Controls.Add(_dataGridViewMatrixC);
            comboBoxSrc.SelectedIndexChanged += ValueChanged;
            comboBoxDest.SelectedIndexChanged += ValueChanged;
            comboBoxCache.SelectedIndexChanged += ValueChanged;
            numericUpDownHeightA.ValueChanged += ValueChanged;
            numericUpDownHeightB.ValueChanged += ValueChanged;
            numericUpDownWidthA.ValueChanged += ValueChanged;
            numericUpDownWidthB.ValueChanged += ValueChanged;
            numericUpDownBlocks1.ValueChanged += ValueChanged;
            numericUpDownBlocks2.ValueChanged += ValueChanged;
            numericUpDownThreads1.ValueChanged += ValueChanged;
            numericUpDownThreads2.ValueChanged += ValueChanged;
            radioButtonPlus.CheckedChanged += ValueChanged;
            radioButtonMinus.CheckedChanged += ValueChanged;
            radioButtonProduct.CheckedChanged += ValueChanged;
            radioButtonRotate.CheckedChanged += ValueChanged;
        }

        public int HeightA
        {
            get { return Convert.ToInt32(numericUpDownHeightA.Value); }
            set { numericUpDownHeightA.Value = value; }
        }

        public int HeightB
        {
            get { return Convert.ToInt32(numericUpDownHeightB.Value); }
            set { numericUpDownHeightB.Value = value; }
        }

        public int HeightC
        {
            get { return Convert.ToInt32(numericUpDownHeightC.Value); }
            set { numericUpDownHeightC.Value = value; }
        }

        public int WidthA
        {
            get { return Convert.ToInt32(numericUpDownWidthA.Value); }
            set { numericUpDownWidthA.Value = value; }
        }

        public int WidthB
        {
            get { return Convert.ToInt32(numericUpDownWidthB.Value); }
            set { numericUpDownWidthB.Value = value; }
        }

        public int WidthC
        {
            get { return Convert.ToInt32(numericUpDownWidthC.Value); }
            set { numericUpDownWidthC.Value = value; }
        }

        public int Blocks1
        {
            get { return Convert.ToInt32(numericUpDownBlocks1.Value); }
            set { numericUpDownBlocks1.Value = value; }
        }

        public int Blocks2
        {
            get { return Convert.ToInt32(numericUpDownBlocks2.Value); }
            set { numericUpDownBlocks2.Value = value; }
        }

        public int Threads1
        {
            get { return Convert.ToInt32(numericUpDownThreads1.Value); }
            set { numericUpDownThreads1.Value = value; }
        }

        public int Threads2
        {
            get { return Convert.ToInt32(numericUpDownThreads2.Value); }
            set { numericUpDownThreads2.Value = value; }
        }

        public bool OperatorPlus
        {
            get { return radioButtonPlus.Checked; }
            set { radioButtonPlus.Checked = value; }
        }

        public bool OperatorMinus
        {
            get { return radioButtonMinus.Checked; }
            set { radioButtonMinus.Checked = value; }
        }

        public bool OperatorProduct
        {
            get { return radioButtonProduct.Checked; }
            set { radioButtonProduct.Checked = value; }
        }

        public bool OperatorRotate
        {
            get { return radioButtonRotate.Checked; }
            set { radioButtonRotate.Checked = value; }
        }

        public string SrcMemory
        {
            get { return comboBoxSrc.Text; }
            set { comboBoxSrc.Text = value; }
        }

        public string DestMemory
        {
            get { return comboBoxDest.Text; }
            set { comboBoxDest.Text = value; }
        }

        public string CacheMemory
        {
            get { return comboBoxCache.Text; }
            set { comboBoxCache.Text = value; }
        }

        public string OpCode
        {
            get
            {
                if (OperatorPlus) return "+";
                if (OperatorMinus) return "-";
                if (OperatorProduct) return "*";
                return "";
            }
        }

        private void ValueChanged(object sender, EventArgs e)
        {
            var sb = new StringBuilder();
            sb.AppendLine("show info");
            sb.AppendLine("read a \"" + Path.GetTempPath() + "MATRIX_A.csv\"");
            if(!OperatorRotate) sb.AppendLine("read b \"" + Path.GetTempPath() + "MATRIX_B.csv\"");
            sb.AppendLine("use src " + SrcMemory);
            sb.AppendLine("use dest " + DestMemory);
            sb.AppendLine("use cache " + CacheMemory);
            sb.AppendLine("set blocks " + Blocks1 + " " + Blocks2);
            sb.AppendLine("set threads " + Threads1 + " " + Threads2);
            if (!OperatorRotate) sb.AppendLine("let c = a " + OpCode + " b");
            if (OperatorRotate) sb.AppendLine("let b = rot a");
            if (!OperatorRotate) sb.AppendLine("write c \"" + Path.GetTempPath() + "MATRIX_C.csv\"");
            if (OperatorRotate) sb.AppendLine("write b \"" + Path.GetTempPath() + "MATRIX_B.csv\"");
            sb.AppendLine("free a");
            sb.AppendLine("free b");
            if (!OperatorRotate) sb.AppendLine("free c");
            textBoxScript.Text = sb.ToString();

            if (OperatorRotate)
            {
                WidthB = HeightA;
                HeightB = WidthA;
            }
            else
            {
                HeightC = HeightA;
                WidthC = WidthB;
            }

            _dataGridViewMatrixA.ResizeOurself(HeightA, WidthA);
            _dataGridViewMatrixB.ResizeOurself(HeightB, WidthB);
            _dataGridViewMatrixC.ResizeOurself(HeightC, WidthC);
        }

        public async void Execute()
        {
            string fileNameLog = Path.GetTempPath() + "MATRIX.log";
            string fileNameA = Path.GetTempPath() + "MATRIX_A.csv";
            string fileNameB = Path.GetTempPath() + "MATRIX_B.csv";
            string fileNameC = Path.GetTempPath() + "MATRIX_C.csv";
            string fileNameScript = Path.GetTempPath() + "MATRIX.SCRIPT";
            using (var writer = new StreamWriter(File.Open(fileNameScript, FileMode.Create)))
            {
                await writer.WriteAsync(textBoxScript.Text);
            }
            string[,] matrix = _dataGridViewMatrixA.TheData;
            using (var writer = new StreamWriter(File.Open(fileNameA, FileMode.Create)))
            {
                for (int i = 0; i < matrix.GetLength(0); i++)
                    for (int j = 0; j < matrix.GetLength(1); j++)
                    {
                        await writer.WriteAsync(matrix[i, j]);
                        if (j < matrix.GetLength(1) - 1) await writer.WriteAsync(";");
                        else await writer.WriteLineAsync();
                    }
                writer.Close();
            }
            if (!OperatorRotate)
            {
                matrix = _dataGridViewMatrixB.TheData;
                using (var writer = new StreamWriter(File.Open(fileNameB, FileMode.Create)))
                {
                    for (int i = 0; i < matrix.GetLength(0); i++)
                        for (int j = 0; j < matrix.GetLength(1); j++)
                        {
                            await writer.WriteAsync(matrix[i, j]);
                            if (j < matrix.GetLength(1) - 1) await writer.WriteAsync(";");
                            else await writer.WriteLineAsync();
                        }
                    writer.Close();
                }
            }
            string command = string.Format("/C cudamatrix.exe < \"{0}\" > \"{1}\"", fileNameScript, fileNameLog);
            Debug.WriteLine(command);

            DateTime start = DateTime.Now;
            Process process = Process.Start("cmd", command);

            if (process == null) return;
            process.WaitForExit();

            DateTime end = DateTime.Now;

            var regex = new Regex(@"\s*(?<data>[-]?\d*(\.\d*)?)\s*([;]|\Z)");
            if (OperatorRotate)
            {
                matrix = new string[HeightB, WidthB];
                using (var reader = new StreamReader(File.Open(fileNameB, FileMode.Open)))
                {
                    int row = 0;
                    for (string line = await reader.ReadLineAsync(); ;
                        line = await reader.ReadLineAsync())
                    {
                        int col = 0;
                        foreach (
                            Match match in
                                regex.Matches(line)
                                    .Cast<Match>()
                                    .Where(match => row < matrix.GetLength(0) && col < matrix.GetLength(1)))
                            matrix[row, col++] = match.Groups["data"].Value;
                        if (reader.EndOfStream) break;
                        row++;
                    }
                    reader.Close();
                }
                _dataGridViewMatrixB.TheData = matrix;
            }
            if (!OperatorRotate)
            {
                matrix = new string[HeightC, WidthC];
                using (var reader = new StreamReader(File.Open(fileNameC, FileMode.Open)))
                {
                    int row = 0;
                    for (string line = await reader.ReadLineAsync(); ;
                        line = await reader.ReadLineAsync())
                    {
                        int col = 0;
                        foreach (
                            Match match in
                                regex.Matches(line)
                                    .Cast<Match>()
                                    .Where(match => row < matrix.GetLength(0) && col < matrix.GetLength(1)))
                            matrix[row, col++] = match.Groups["data"].Value;
                        if (reader.EndOfStream) break;
                        row++;
                    }
                    reader.Close();
                }
                _dataGridViewMatrixC.TheData = matrix;
            }
            using (var reader = new StreamReader(File.Open(fileNameLog, FileMode.Open)))
            {
                textBoxLog.Text = await reader.ReadToEndAsync();
            }

            var timeSpan = new TimeSpan(end.Ticks - start.Ticks);
            MessageBox.Show(timeSpan.ToString());
        }

        public void Random(double minimum, double maximum)
        {
            var matrix = new string[HeightA, WidthA];
            for (int i = 0; i < matrix.GetLength(0); i++)
                for (int j = 0; j < matrix.GetLength(1); j++)
                {
                    matrix[i, j] =Convert.ToInt16(minimum + (maximum - minimum) * Rnd.NextDouble()).ToString(CultureInfo.InvariantCulture);
                }
            _dataGridViewMatrixA.TheData = matrix;
            matrix = new string[HeightB, WidthB];
            for (int i = 0; i < matrix.GetLength(0); i++)
                for (int j = 0; j < matrix.GetLength(1); j++)
                {
                    matrix[i, j] =Convert.ToInt16(minimum + (maximum - minimum)*Rnd.NextDouble()).ToString(CultureInfo.InvariantCulture);
                }
            _dataGridViewMatrixB.TheData = matrix;
        }
    }
}