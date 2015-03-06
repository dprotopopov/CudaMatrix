using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace Matrix.Editor
{
    public partial class CudaEditorForm : Form, EditorForm
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

        public CudaEditorForm()
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

        private string Script
        {
            get { return textBoxScript.Text; }
            set { textBoxScript.Text = value; }
        }

        public async void Execute()
        {
            string fileNameLog = "MATRIX.log";
            string fileNameA = "MATRIX_A.csv";
            string fileNameB = "MATRIX_B.csv";
            string fileNameC = "MATRIX_C.csv";
            string fileNameScript = "MATRIX.SCRIPT";
            using (var writer = new StreamWriter(File.Open(fileNameScript, FileMode.Create)))
            {
                await writer.WriteAsync(Script);
            }
            WriteCsvMatrix(fileNameA, _dataGridViewMatrixA.TheData);
            if (!OperatorRotate) WriteCsvMatrix(fileNameB, _dataGridViewMatrixB.TheData);

            string command = string.Format("/C cudamatrix.exe < \"{0}\" > \"{1}\"", fileNameScript, fileNameLog);
            Debug.WriteLine(command);

            DateTime start = DateTime.Now;
            Process process = Process.Start("cmd", command);

            if (process == null) return;
            process.WaitForExit();

            DateTime end = DateTime.Now;

            if (OperatorRotate) _dataGridViewMatrixB.TheData = await ReadCsvMatrix(fileNameB);
            if (!OperatorRotate) _dataGridViewMatrixC.TheData = await ReadCsvMatrix(fileNameC);
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
                    matrix[i, j] =
                        Convert.ToInt16(minimum + (maximum - minimum)*Rnd.NextDouble())
                            .ToString(CultureInfo.InvariantCulture);
                }
            _dataGridViewMatrixA.TheData = matrix;
            matrix = new string[HeightB, WidthB];
            for (int i = 0; i < matrix.GetLength(0); i++)
                for (int j = 0; j < matrix.GetLength(1); j++)
                {
                    matrix[i, j] =
                        Convert.ToInt16(minimum + (maximum - minimum)*Rnd.NextDouble())
                            .ToString(CultureInfo.InvariantCulture);
                }
            _dataGridViewMatrixB.TheData = matrix;
        }

        private void ValueChanged(object sender, EventArgs e)
        {
            var sb = new StringBuilder();
            sb.AppendLine("show info");
            sb.AppendLine("read a \"MATRIX_A.csv\"");
            if (!OperatorRotate) sb.AppendLine("read b \"MATRIX_B.csv\"");
            sb.AppendLine("use src " + SrcMemory);
            sb.AppendLine("use dest " + DestMemory);
            sb.AppendLine("use cache " + CacheMemory);
            sb.AppendLine("set blocks " + Blocks1 + " " + Blocks2);
            sb.AppendLine("set threads " + Threads1 + " " + Threads2);
            if (!OperatorRotate) sb.AppendLine("let c = a " + OpCode + " b");
            if (OperatorRotate) sb.AppendLine("let b = rot a");
            if (!OperatorRotate) sb.AppendLine("write c \"MATRIX_C.csv\"");
            if (OperatorRotate) sb.AppendLine("write b \"MATRIX_B.csv\"");
            sb.AppendLine("free a");
            sb.AppendLine("free b");
            if (!OperatorRotate) sb.AppendLine("free c");

            Script = sb.ToString();

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

        public static async Task<string[,]> ReadCsvMatrix(string fileName)
        {
            var lists = new List<List<string>>();
            using (var reader = new StreamReader(File.Open(fileName, FileMode.Open)))
            {
                for (string line = await reader.ReadLineAsync();;
                    line = await reader.ReadLineAsync())
                {
                    lists.Add(line.Split(';').ToList());
                    if (reader.EndOfStream) break;
                }
                reader.Close();
                var matrix = new string[lists.Count, lists.Max(item => item.Count)];
                for (int i = 0; i < matrix.GetLength(0); i++)
                    for (int j = 0; j < matrix.GetLength(1); j++)
                        matrix[i, j] = lists[i][j];
                return matrix;
            }
        }

        public static async void WriteCsvMatrix(string fileName, string[,] matrix)
        {
            using (var writer = new StreamWriter(File.Open(fileName, FileMode.Create)))
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
    }
}