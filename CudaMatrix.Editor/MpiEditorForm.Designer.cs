namespace Matrix.Editor
{
    partial class MpiEditorForm
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.splitContainer1 = new System.Windows.Forms.SplitContainer();
            this.splitContainer2 = new System.Windows.Forms.SplitContainer();
            this.groupBox2 = new System.Windows.Forms.GroupBox();
            this.groupBox3 = new System.Windows.Forms.GroupBox();
            this.label1 = new System.Windows.Forms.Label();
            this.numericUpDownNumberOfProcess = new System.Windows.Forms.NumericUpDown();
            this.groupBox5 = new System.Windows.Forms.GroupBox();
            this.comboBoxOpCode = new System.Windows.Forms.ComboBox();
            this.groupBox4 = new System.Windows.Forms.GroupBox();
            this.label11 = new System.Windows.Forms.Label();
            this.label10 = new System.Windows.Forms.Label();
            this.label9 = new System.Windows.Forms.Label();
            this.numericUpDownWidthC = new System.Windows.Forms.NumericUpDown();
            this.numericUpDownHeightC = new System.Windows.Forms.NumericUpDown();
            this.numericUpDownWidthB = new System.Windows.Forms.NumericUpDown();
            this.numericUpDownHeightB = new System.Windows.Forms.NumericUpDown();
            this.numericUpDownWidthA = new System.Windows.Forms.NumericUpDown();
            this.numericUpDownHeightA = new System.Windows.Forms.NumericUpDown();
            this.tabControl1 = new System.Windows.Forms.TabControl();
            this.tabPage1 = new System.Windows.Forms.TabPage();
            this.textBoxScript = new System.Windows.Forms.TextBox();
            this.tabPageA = new System.Windows.Forms.TabPage();
            this.tabPageB = new System.Windows.Forms.TabPage();
            this.tabPageC = new System.Windows.Forms.TabPage();
            this.groupBox1 = new System.Windows.Forms.GroupBox();
            this.textBoxLog = new System.Windows.Forms.TextBox();
            this.label2 = new System.Windows.Forms.Label();
            this.textBoxWorkingDir = new System.Windows.Forms.TextBox();
            ((System.ComponentModel.ISupportInitialize)(this.splitContainer1)).BeginInit();
            this.splitContainer1.Panel1.SuspendLayout();
            this.splitContainer1.Panel2.SuspendLayout();
            this.splitContainer1.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.splitContainer2)).BeginInit();
            this.splitContainer2.Panel1.SuspendLayout();
            this.splitContainer2.Panel2.SuspendLayout();
            this.splitContainer2.SuspendLayout();
            this.groupBox2.SuspendLayout();
            this.groupBox3.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDownNumberOfProcess)).BeginInit();
            this.groupBox5.SuspendLayout();
            this.groupBox4.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDownWidthC)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDownHeightC)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDownWidthB)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDownHeightB)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDownWidthA)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDownHeightA)).BeginInit();
            this.tabControl1.SuspendLayout();
            this.tabPage1.SuspendLayout();
            this.groupBox1.SuspendLayout();
            this.SuspendLayout();
            // 
            // splitContainer1
            // 
            this.splitContainer1.Dock = System.Windows.Forms.DockStyle.Fill;
            this.splitContainer1.Location = new System.Drawing.Point(0, 0);
            this.splitContainer1.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.splitContainer1.Name = "splitContainer1";
            this.splitContainer1.Orientation = System.Windows.Forms.Orientation.Horizontal;
            // 
            // splitContainer1.Panel1
            // 
            this.splitContainer1.Panel1.Controls.Add(this.splitContainer2);
            // 
            // splitContainer1.Panel2
            // 
            this.splitContainer1.Panel2.Controls.Add(this.groupBox1);
            this.splitContainer1.Size = new System.Drawing.Size(1200, 692);
            this.splitContainer1.SplitterDistance = 551;
            this.splitContainer1.SplitterWidth = 5;
            this.splitContainer1.TabIndex = 0;
            // 
            // splitContainer2
            // 
            this.splitContainer2.Dock = System.Windows.Forms.DockStyle.Fill;
            this.splitContainer2.Location = new System.Drawing.Point(0, 0);
            this.splitContainer2.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.splitContainer2.Name = "splitContainer2";
            // 
            // splitContainer2.Panel1
            // 
            this.splitContainer2.Panel1.Controls.Add(this.groupBox2);
            // 
            // splitContainer2.Panel2
            // 
            this.splitContainer2.Panel2.Controls.Add(this.tabControl1);
            this.splitContainer2.Size = new System.Drawing.Size(1200, 551);
            this.splitContainer2.SplitterDistance = 528;
            this.splitContainer2.TabIndex = 0;
            // 
            // groupBox2
            // 
            this.groupBox2.Controls.Add(this.groupBox3);
            this.groupBox2.Controls.Add(this.groupBox5);
            this.groupBox2.Controls.Add(this.groupBox4);
            this.groupBox2.Dock = System.Windows.Forms.DockStyle.Fill;
            this.groupBox2.Location = new System.Drawing.Point(0, 0);
            this.groupBox2.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.groupBox2.Name = "groupBox2";
            this.groupBox2.Padding = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.groupBox2.Size = new System.Drawing.Size(528, 551);
            this.groupBox2.TabIndex = 0;
            this.groupBox2.TabStop = false;
            this.groupBox2.Text = "Options";
            // 
            // groupBox3
            // 
            this.groupBox3.Controls.Add(this.textBoxWorkingDir);
            this.groupBox3.Controls.Add(this.label2);
            this.groupBox3.Controls.Add(this.label1);
            this.groupBox3.Controls.Add(this.numericUpDownNumberOfProcess);
            this.groupBox3.Location = new System.Drawing.Point(32, 374);
            this.groupBox3.Name = "groupBox3";
            this.groupBox3.Size = new System.Drawing.Size(366, 154);
            this.groupBox3.TabIndex = 3;
            this.groupBox3.TabStop = false;
            this.groupBox3.Text = "MPI";
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(40, 105);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(147, 20);
            this.label1.TabIndex = 1;
            this.label1.Text = "Number Of Process";
            // 
            // numericUpDownNumberOfProcess
            // 
            this.numericUpDownNumberOfProcess.Location = new System.Drawing.Point(197, 102);
            this.numericUpDownNumberOfProcess.Maximum = new decimal(new int[] {
            1000,
            0,
            0,
            0});
            this.numericUpDownNumberOfProcess.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.numericUpDownNumberOfProcess.Name = "numericUpDownNumberOfProcess";
            this.numericUpDownNumberOfProcess.Size = new System.Drawing.Size(120, 26);
            this.numericUpDownNumberOfProcess.TabIndex = 0;
            this.numericUpDownNumberOfProcess.Value = new decimal(new int[] {
            1,
            0,
            0,
            0});
            // 
            // groupBox5
            // 
            this.groupBox5.Controls.Add(this.comboBoxOpCode);
            this.groupBox5.Location = new System.Drawing.Point(32, 243);
            this.groupBox5.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.groupBox5.Name = "groupBox5";
            this.groupBox5.Padding = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.groupBox5.Size = new System.Drawing.Size(366, 112);
            this.groupBox5.TabIndex = 2;
            this.groupBox5.TabStop = false;
            this.groupBox5.Text = "opCode";
            // 
            // comboBoxOpCode
            // 
            this.comboBoxOpCode.FormattingEnabled = true;
            this.comboBoxOpCode.Items.AddRange(new object[] {
            "add",
            "sub",
            "mul",
            "mtv",
            "det",
            "inv",
            "rot",
            "one",
            "nil",
            "mov"});
            this.comboBoxOpCode.Location = new System.Drawing.Point(80, 46);
            this.comboBoxOpCode.Name = "comboBoxOpCode";
            this.comboBoxOpCode.Size = new System.Drawing.Size(194, 28);
            this.comboBoxOpCode.TabIndex = 0;
            // 
            // groupBox4
            // 
            this.groupBox4.Controls.Add(this.label11);
            this.groupBox4.Controls.Add(this.label10);
            this.groupBox4.Controls.Add(this.label9);
            this.groupBox4.Controls.Add(this.numericUpDownWidthC);
            this.groupBox4.Controls.Add(this.numericUpDownHeightC);
            this.groupBox4.Controls.Add(this.numericUpDownWidthB);
            this.groupBox4.Controls.Add(this.numericUpDownHeightB);
            this.groupBox4.Controls.Add(this.numericUpDownWidthA);
            this.groupBox4.Controls.Add(this.numericUpDownHeightA);
            this.groupBox4.Location = new System.Drawing.Point(32, 31);
            this.groupBox4.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.groupBox4.Name = "groupBox4";
            this.groupBox4.Padding = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.groupBox4.Size = new System.Drawing.Size(366, 204);
            this.groupBox4.TabIndex = 1;
            this.groupBox4.TabStop = false;
            this.groupBox4.Text = "Sizes";
            // 
            // label11
            // 
            this.label11.AutoSize = true;
            this.label11.Location = new System.Drawing.Point(18, 151);
            this.label11.Name = "label11";
            this.label11.Size = new System.Drawing.Size(20, 20);
            this.label11.TabIndex = 8;
            this.label11.Text = "C";
            // 
            // label10
            // 
            this.label10.AutoSize = true;
            this.label10.Location = new System.Drawing.Point(18, 92);
            this.label10.Name = "label10";
            this.label10.Size = new System.Drawing.Size(20, 20);
            this.label10.TabIndex = 7;
            this.label10.Text = "B";
            // 
            // label9
            // 
            this.label9.AutoSize = true;
            this.label9.Location = new System.Drawing.Point(18, 58);
            this.label9.Name = "label9";
            this.label9.Size = new System.Drawing.Size(20, 20);
            this.label9.TabIndex = 6;
            this.label9.Text = "A";
            // 
            // numericUpDownWidthC
            // 
            this.numericUpDownWidthC.Location = new System.Drawing.Point(197, 149);
            this.numericUpDownWidthC.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.numericUpDownWidthC.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.numericUpDownWidthC.Name = "numericUpDownWidthC";
            this.numericUpDownWidthC.ReadOnly = true;
            this.numericUpDownWidthC.Size = new System.Drawing.Size(135, 26);
            this.numericUpDownWidthC.TabIndex = 5;
            this.numericUpDownWidthC.Value = new decimal(new int[] {
            10,
            0,
            0,
            0});
            // 
            // numericUpDownHeightC
            // 
            this.numericUpDownHeightC.Location = new System.Drawing.Point(40, 149);
            this.numericUpDownHeightC.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.numericUpDownHeightC.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.numericUpDownHeightC.Name = "numericUpDownHeightC";
            this.numericUpDownHeightC.ReadOnly = true;
            this.numericUpDownHeightC.Size = new System.Drawing.Size(135, 26);
            this.numericUpDownHeightC.TabIndex = 4;
            this.numericUpDownHeightC.Value = new decimal(new int[] {
            10,
            0,
            0,
            0});
            // 
            // numericUpDownWidthB
            // 
            this.numericUpDownWidthB.Location = new System.Drawing.Point(197, 90);
            this.numericUpDownWidthB.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.numericUpDownWidthB.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.numericUpDownWidthB.Name = "numericUpDownWidthB";
            this.numericUpDownWidthB.Size = new System.Drawing.Size(135, 26);
            this.numericUpDownWidthB.TabIndex = 3;
            this.numericUpDownWidthB.Value = new decimal(new int[] {
            10,
            0,
            0,
            0});
            // 
            // numericUpDownHeightB
            // 
            this.numericUpDownHeightB.Location = new System.Drawing.Point(40, 91);
            this.numericUpDownHeightB.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.numericUpDownHeightB.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.numericUpDownHeightB.Name = "numericUpDownHeightB";
            this.numericUpDownHeightB.Size = new System.Drawing.Size(135, 26);
            this.numericUpDownHeightB.TabIndex = 2;
            this.numericUpDownHeightB.Value = new decimal(new int[] {
            10,
            0,
            0,
            0});
            // 
            // numericUpDownWidthA
            // 
            this.numericUpDownWidthA.Location = new System.Drawing.Point(197, 55);
            this.numericUpDownWidthA.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.numericUpDownWidthA.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.numericUpDownWidthA.Name = "numericUpDownWidthA";
            this.numericUpDownWidthA.Size = new System.Drawing.Size(135, 26);
            this.numericUpDownWidthA.TabIndex = 1;
            this.numericUpDownWidthA.Value = new decimal(new int[] {
            10,
            0,
            0,
            0});
            // 
            // numericUpDownHeightA
            // 
            this.numericUpDownHeightA.Location = new System.Drawing.Point(40, 55);
            this.numericUpDownHeightA.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.numericUpDownHeightA.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.numericUpDownHeightA.Name = "numericUpDownHeightA";
            this.numericUpDownHeightA.Size = new System.Drawing.Size(135, 26);
            this.numericUpDownHeightA.TabIndex = 0;
            this.numericUpDownHeightA.Value = new decimal(new int[] {
            10,
            0,
            0,
            0});
            // 
            // tabControl1
            // 
            this.tabControl1.Controls.Add(this.tabPage1);
            this.tabControl1.Controls.Add(this.tabPageA);
            this.tabControl1.Controls.Add(this.tabPageB);
            this.tabControl1.Controls.Add(this.tabPageC);
            this.tabControl1.Dock = System.Windows.Forms.DockStyle.Fill;
            this.tabControl1.Location = new System.Drawing.Point(0, 0);
            this.tabControl1.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.tabControl1.Name = "tabControl1";
            this.tabControl1.SelectedIndex = 0;
            this.tabControl1.Size = new System.Drawing.Size(668, 551);
            this.tabControl1.TabIndex = 0;
            // 
            // tabPage1
            // 
            this.tabPage1.Controls.Add(this.textBoxScript);
            this.tabPage1.Location = new System.Drawing.Point(4, 29);
            this.tabPage1.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.tabPage1.Name = "tabPage1";
            this.tabPage1.Padding = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.tabPage1.Size = new System.Drawing.Size(660, 518);
            this.tabPage1.TabIndex = 0;
            this.tabPage1.Text = "Script";
            this.tabPage1.UseVisualStyleBackColor = true;
            // 
            // textBoxScript
            // 
            this.textBoxScript.Dock = System.Windows.Forms.DockStyle.Fill;
            this.textBoxScript.Location = new System.Drawing.Point(3, 4);
            this.textBoxScript.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.textBoxScript.Multiline = true;
            this.textBoxScript.Name = "textBoxScript";
            this.textBoxScript.Size = new System.Drawing.Size(654, 510);
            this.textBoxScript.TabIndex = 0;
            // 
            // tabPageA
            // 
            this.tabPageA.Location = new System.Drawing.Point(4, 29);
            this.tabPageA.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.tabPageA.Name = "tabPageA";
            this.tabPageA.Padding = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.tabPageA.Size = new System.Drawing.Size(660, 518);
            this.tabPageA.TabIndex = 1;
            this.tabPageA.Text = "A";
            this.tabPageA.UseVisualStyleBackColor = true;
            // 
            // tabPageB
            // 
            this.tabPageB.Location = new System.Drawing.Point(4, 29);
            this.tabPageB.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.tabPageB.Name = "tabPageB";
            this.tabPageB.Size = new System.Drawing.Size(660, 518);
            this.tabPageB.TabIndex = 2;
            this.tabPageB.Text = "B";
            this.tabPageB.UseVisualStyleBackColor = true;
            // 
            // tabPageC
            // 
            this.tabPageC.Location = new System.Drawing.Point(4, 29);
            this.tabPageC.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.tabPageC.Name = "tabPageC";
            this.tabPageC.Size = new System.Drawing.Size(660, 518);
            this.tabPageC.TabIndex = 3;
            this.tabPageC.Text = "C";
            this.tabPageC.UseVisualStyleBackColor = true;
            // 
            // groupBox1
            // 
            this.groupBox1.Controls.Add(this.textBoxLog);
            this.groupBox1.Dock = System.Windows.Forms.DockStyle.Fill;
            this.groupBox1.Location = new System.Drawing.Point(0, 0);
            this.groupBox1.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.Padding = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.groupBox1.Size = new System.Drawing.Size(1200, 136);
            this.groupBox1.TabIndex = 0;
            this.groupBox1.TabStop = false;
            this.groupBox1.Text = "Logging";
            // 
            // textBoxLog
            // 
            this.textBoxLog.Dock = System.Windows.Forms.DockStyle.Fill;
            this.textBoxLog.Location = new System.Drawing.Point(3, 23);
            this.textBoxLog.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.textBoxLog.Multiline = true;
            this.textBoxLog.Name = "textBoxLog";
            this.textBoxLog.ScrollBars = System.Windows.Forms.ScrollBars.Both;
            this.textBoxLog.Size = new System.Drawing.Size(1194, 109);
            this.textBoxLog.TabIndex = 0;
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(40, 58);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(88, 20);
            this.label2.TabIndex = 2;
            this.label2.Text = "Working dir";
            // 
            // textBoxWorkingDir
            // 
            this.textBoxWorkingDir.Location = new System.Drawing.Point(197, 58);
            this.textBoxWorkingDir.Name = "textBoxWorkingDir";
            this.textBoxWorkingDir.Size = new System.Drawing.Size(120, 26);
            this.textBoxWorkingDir.TabIndex = 3;
            this.textBoxWorkingDir.Text = "T:\\";
            // 
            // MpiEditorForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(9F, 20F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(1200, 692);
            this.Controls.Add(this.splitContainer1);
            this.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.Name = "MpiEditorForm";
            this.Text = "MatrixForm";
            this.WindowState = System.Windows.Forms.FormWindowState.Maximized;
            this.splitContainer1.Panel1.ResumeLayout(false);
            this.splitContainer1.Panel2.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.splitContainer1)).EndInit();
            this.splitContainer1.ResumeLayout(false);
            this.splitContainer2.Panel1.ResumeLayout(false);
            this.splitContainer2.Panel2.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.splitContainer2)).EndInit();
            this.splitContainer2.ResumeLayout(false);
            this.groupBox2.ResumeLayout(false);
            this.groupBox3.ResumeLayout(false);
            this.groupBox3.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDownNumberOfProcess)).EndInit();
            this.groupBox5.ResumeLayout(false);
            this.groupBox4.ResumeLayout(false);
            this.groupBox4.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDownWidthC)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDownHeightC)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDownWidthB)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDownHeightB)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDownWidthA)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDownHeightA)).EndInit();
            this.tabControl1.ResumeLayout(false);
            this.tabPage1.ResumeLayout(false);
            this.tabPage1.PerformLayout();
            this.groupBox1.ResumeLayout(false);
            this.groupBox1.PerformLayout();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.SplitContainer splitContainer1;
        private System.Windows.Forms.SplitContainer splitContainer2;
        private System.Windows.Forms.GroupBox groupBox2;
        private System.Windows.Forms.GroupBox groupBox5;
        private System.Windows.Forms.GroupBox groupBox4;
        private System.Windows.Forms.NumericUpDown numericUpDownWidthC;
        private System.Windows.Forms.NumericUpDown numericUpDownHeightC;
        private System.Windows.Forms.NumericUpDown numericUpDownWidthB;
        private System.Windows.Forms.NumericUpDown numericUpDownHeightB;
        private System.Windows.Forms.NumericUpDown numericUpDownWidthA;
        private System.Windows.Forms.NumericUpDown numericUpDownHeightA;
        private System.Windows.Forms.TabControl tabControl1;
        private System.Windows.Forms.TabPage tabPage1;
        private System.Windows.Forms.TabPage tabPageA;
        private System.Windows.Forms.GroupBox groupBox1;
        private System.Windows.Forms.TextBox textBoxLog;
        private System.Windows.Forms.TextBox textBoxScript;
        private System.Windows.Forms.TabPage tabPageB;
        private System.Windows.Forms.TabPage tabPageC;
        private System.Windows.Forms.Label label11;
        private System.Windows.Forms.Label label10;
        private System.Windows.Forms.Label label9;
        private System.Windows.Forms.ComboBox comboBoxOpCode;
        private System.Windows.Forms.GroupBox groupBox3;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.NumericUpDown numericUpDownNumberOfProcess;
        private System.Windows.Forms.TextBox textBoxWorkingDir;
        private System.Windows.Forms.Label label2;
    }
}