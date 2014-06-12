using System;
using System.Windows.Forms;

namespace CudaMatrix.Editor
{
    public partial class MainForm : Form
    {
        static readonly RandomDialog RandomDialog = new RandomDialog();
        public MainForm()
        {
            InitializeComponent();
        }

        private void exitToolStripMenuItem_Click(object sender, EventArgs e)
        {
            Close();
        }

        private void newToolStripMenuItem_Click(object sender, EventArgs e)
        {
            var child = new MatrixForm
            {
                MdiParent = this
            };
            child.Show();
        }

        private void executeToolStripMenuItem_Click(object sender, EventArgs e)
        {
            var child = ActiveMdiChild as MatrixForm;
            if (child == null) return;
            child.Execute();
        }

        private void randomToolStripMenuItem_Click(object sender, EventArgs e)
        {
            var child = ActiveMdiChild as MatrixForm;
            if (child == null) return;
            if (RandomDialog.ShowDialog() != DialogResult.OK) return;
            child.Random(RandomDialog.Minimum,RandomDialog.Maximum);
        }
    }
}