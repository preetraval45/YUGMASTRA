import { jsPDF } from 'jspdf';
import autoTable from 'jspdf-autotable';
import Papa from 'papaparse';

export interface ExportData {
  headers: string[];
  rows: any[][];
  title?: string;
  filename?: string;
}

export function exportToCSV(data: ExportData): void {
  const csv = Papa.unparse({
    fields: data.headers,
    data: data.rows,
  });

  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
  const link = document.createElement('a');
  const url = URL.createObjectURL(blob);

  link.setAttribute('href', url);
  link.setAttribute('download', data.filename || 'export.csv');
  link.style.visibility = 'hidden';

  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

export function exportToPDF(data: ExportData): void {
  const doc = new jsPDF();

  // Add title
  if (data.title) {
    doc.setFontSize(18);
    doc.text(data.title, 14, 22);
  }

  // Add table
  autoTable(doc, {
    head: [data.headers],
    body: data.rows,
    startY: data.title ? 30 : 10,
    theme: 'striped',
    headStyles: {
      fillColor: [59, 130, 246], // Blue
      textColor: [255, 255, 255],
      fontStyle: 'bold',
    },
    styles: {
      fontSize: 10,
      cellPadding: 3,
    },
    alternateRowStyles: {
      fillColor: [245, 247, 250],
    },
  });

  // Add footer with timestamp
  const pageCount = (doc as any).internal.getNumberOfPages();
  for (let i = 1; i <= pageCount; i++) {
    doc.setPage(i);
    doc.setFontSize(8);
    doc.text(
      `Generated: ${new Date().toLocaleString()} | Page ${i} of ${pageCount}`,
      14,
      doc.internal.pageSize.height - 10
    );
  }

  doc.save(data.filename || 'export.pdf');
}

export function exportToJSON(data: any, filename: string = 'export.json'): void {
  const json = JSON.stringify(data, null, 2);
  const blob = new Blob([json], { type: 'application/json' });
  const link = document.createElement('a');
  const url = URL.createObjectURL(blob);

  link.setAttribute('href', url);
  link.setAttribute('download', filename);
  link.style.visibility = 'hidden';

  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

// Battle report export
export function exportBattleReport(battle: any): void {
  const data: ExportData = {
    title: `Battle Report - ${battle.id}`,
    filename: `battle-report-${battle.id}-${Date.now()}.pdf`,
    headers: ['Metric', 'Value'],
    rows: [
      ['Battle ID', battle.id],
      ['Status', battle.status],
      ['Red Team Score', battle.redScore.toString()],
      ['Blue Team Score', battle.blueScore.toString()],
      ['Duration', `${battle.duration}s`],
      ['Total Attacks', battle.metrics?.totalAttacks || 0],
      ['Successful Attacks', battle.metrics?.successfulAttacks || 0],
      ['Detected Attacks', battle.metrics?.detectedAttacks || 0],
      ['Total Defenses', battle.metrics?.totalDefenses || 0],
      ['Effective Defenses', battle.metrics?.effectiveDefenses || 0],
      ['Nash Equilibrium', battle.nashEquilibrium?.toFixed(4) || 'N/A'],
      ['Generation', battle.coevolutionGen.toString()],
      ['Created', new Date(battle.createdAt).toLocaleString()],
    ],
  };

  exportToPDF(data);
}

// Attacks export
export function exportAttacks(attacks: any[]): void {
  const data: ExportData = {
    title: 'Attacks Report',
    filename: `attacks-report-${Date.now()}.csv`,
    headers: [
      'ID',
      'Type',
      'Technique',
      'Target',
      'Severity',
      'Success',
      'Detected',
      'Impact',
      'Timestamp',
    ],
    rows: attacks.map((attack) => [
      attack.id,
      attack.type,
      attack.technique,
      attack.target,
      attack.severity,
      attack.success ? 'Yes' : 'No',
      attack.detected ? 'Yes' : 'No',
      (attack.impact * 100).toFixed(1) + '%',
      new Date(attack.timestamp).toLocaleString(),
    ]),
  };

  exportToCSV(data);
}

// Defenses export
export function exportDefenses(defenses: any[]): void {
  const data: ExportData = {
    title: 'Defenses Report',
    filename: `defenses-report-${Date.now()}.csv`,
    headers: [
      'ID',
      'Action',
      'Rule Type',
      'Effectiveness',
      'Attacks Blocked',
      'False Positives',
      'Timestamp',
    ],
    rows: defenses.map((defense) => [
      defense.id,
      defense.action,
      defense.ruleType,
      (defense.effectiveness * 100).toFixed(1) + '%',
      defense.attacksBlocked.toString(),
      defense.falsePositives.toString(),
      new Date(defense.timestamp).toLocaleString(),
    ]),
  };

  exportToCSV(data);
}
