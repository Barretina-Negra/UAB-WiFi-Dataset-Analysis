import { readdir, readFile } from 'fs/promises';
import { join } from 'path';

// Types
interface APRecord {
  name?: string;
  macaddr?: string;
  client_count?: number;
  cpu_utilization?: number;
  mem_free?: number;
  mem_total?: number;
  group_name?: string;
  site?: string;
  radios?: Array<{
    band?: number;
    utilization?: number;
    channel?: string;
    radio_name?: string;
  }>;
}

interface ConflictiveAP {
  name: string;
  macaddr: string;
  group_code: string;
  conflictivity: number;
  airtime_score: number;
  airtime_score_2g: number;
  airtime_score_5g: number;
  util_2g: number;
  util_5g: number;
  client_count: number;
  client_score: number;
  cpu_utilization: number;
  cpu_score: number;
  mem_used_pct: number;
  mem_score: number;
  breakdown: {
    airtime_contrib: number;
    client_contrib: number;
    cpu_contrib: number;
    mem_contrib: number;
  };
}

type BandMode = 'worst' | 'avg' | '2.4GHz' | '5GHz';

// Utility functions
function clamp(x: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, x));
}

function airtimeScore(util: number, band: '2g' | '5g'): number {
  const u = clamp(util || 0.0, 0.0, 100.0);
  
  if (band === '2g') {
    // 2.4 GHz thresholds
    if (u <= 10) {
      return 0.05 * (u / 10.0); // 0–0.05
    }
    if (u <= 25) {
      return 0.05 + 0.35 * ((u - 10) / 15.0); // 0.05–0.40
    }
    if (u <= 50) {
      return 0.40 + 0.35 * ((u - 25) / 25.0); // 0.40–0.75
    }
    return 0.75 + 0.25 * ((u - 50) / 50.0); // 0.75–1.00
  } else {
    // 5 GHz thresholds
    if (u <= 15) {
      return 0.05 * (u / 15.0); // 0–0.05
    }
    if (u <= 35) {
      return 0.05 + 0.35 * ((u - 15) / 20.0); // 0.05–0.40
    }
    if (u <= 65) {
      return 0.40 + 0.35 * ((u - 35) / 30.0); // 0.40–0.75
    }
    return 0.75 + 0.25 * ((u - 65) / 35.0); // 0.75–1.00
  }
}

function clientPressureScore(nClients: number, peersP95: number): number {
  const n = Math.max(0.0, nClients || 0.0);
  const denom = Math.max(1.0, peersP95 || 1.0);
  const x = Math.log1p(n) / Math.log1p(denom); // 0–1
  return clamp(x, 0.0, 1.0);
}

function cpuHealthScore(cpuPct: number): number {
  const c = clamp(cpuPct || 0.0, 0.0, 100.0);
  if (c <= 70) {
    return 0.0;
  }
  if (c <= 90) {
    return 0.6 * ((c - 70) / 20.0); // 0–0.6
  }
  return 0.6 + 0.4 * ((c - 90) / 10.0); // 0.6–1.0
}

function memHealthScore(memUsedPct: number): number {
  const m = clamp(memUsedPct || 0.0, 0.0, 100.0);
  if (m <= 80) {
    return 0.0;
  }
  if (m <= 95) {
    return 0.6 * ((m - 80) / 15.0); // 0–0.6
  }
  return 0.6 + 0.4 * ((m - 95) / 5.0); // 0.6–1.0
}

function extractGroup(apName: string | undefined): string {
  if (!apName) return 'UNKNOWN';
  const match = apName.match(/^AP-([A-Za-z]+)/);
  return match ? match[1] : 'UNKNOWN';
}

function parseHourFromFilename(filename: string): number | null {
  // Format: AP-info-v2-2025-04-03T00_00_01+02_00.json
  // Extract hour from timestamp
  const match = filename.match(/(\d{4})-(\d{2})-(\d{2})T(\d{2})_(\d{2})_(\d{2})/);
  if (match) {
    return parseInt(match[4], 10);
  }
  return null;
}

function parseHourFromClientFilename(filename: string): number | null {
  // Format: client-info-2025-04-03T00_01_15+02_00-783.json
  const match = filename.match(/client-info-(\d{4})-(\d{2})-(\d{2})T(\d{2})_(\d{2})_(\d{2})/);
  if (match) {
    return parseInt(match[4], 10);
  }
  return null;
}

function calculatePercentile(values: number[], percentile: number): number {
  if (values.length === 0) return 1.0;
  const sorted = [...values].sort((a, b) => a - b);
  const index = Math.ceil((percentile / 100) * sorted.length) - 1;
  return sorted[Math.max(0, index)];
}

async function findFilesByHour(
  apDir: string,
  clientsDir: string,
  targetHour: number
): Promise<{ apFiles: string[]; clientFiles: string[] }> {
  const apFiles: string[] = [];
  const clientFiles: string[] = [];

  // Find AP files
  const apAllFiles = await readdir(apDir);
  for (const file of apAllFiles) {
    if (file.endsWith('.json') && file.startsWith('AP-info-v2-')) {
      const hour = parseHourFromFilename(file);
      if (hour === targetHour) {
        apFiles.push(file);
      }
    }
  }

  // Find client files
  const clientAllFiles = await readdir(clientsDir);
  for (const file of clientAllFiles) {
    if (file.endsWith('.json') && file.startsWith('client-info-')) {
      const hour = parseHourFromClientFilename(file);
      if (hour === targetHour) {
        clientFiles.push(file);
      }
    }
  }

  return { apFiles, clientFiles };
}

async function readAPSnapshot(
  filePath: string,
  bandMode: BandMode = 'worst'
): Promise<APRecord[]> {
  const content = await readFile(filePath, 'utf-8');
  return JSON.parse(content);
}

function processAPs(
  apData: APRecord[],
  bandMode: BandMode = 'worst'
): ConflictiveAP[] {
  const results: ConflictiveAP[] = [];
  const clientCounts: number[] = [];

  // First pass: collect client counts for percentile calculation
  for (const ap of apData) {
    const clientCount = ap.client_count || 0;
    clientCounts.push(clientCount);
  }

  const p95Clients = calculatePercentile(clientCounts, 95);

  // Second pass: calculate conflictivity for each AP
  for (const ap of apData) {
    const name = ap.name || 'UNKNOWN';
    const macaddr = ap.macaddr || 'UNKNOWN';
    const groupCode = extractGroup(name);
    const clientCount = ap.client_count || 0;
    const cpuUtil = ap.cpu_utilization || 0;
    const memFree = ap.mem_free || 0;
    const memTotal = ap.mem_total || 0;

    // Extract radio utilizations
    const util2g: number[] = [];
    const util5g: number[] = [];
    const radios = ap.radios || [];

    for (const radio of radios) {
      const util = radio.utilization;
      const band = radio.band;
      if (util === undefined || util === null) continue;

      if (band === 0) {
        util2g.push(util);
      } else if (band === 1) {
        util5g.push(util);
      }
    }

    const max2g = util2g.length > 0 ? Math.max(...util2g) : NaN;
    const max5g = util5g.length > 0 ? Math.max(...util5g) : NaN;

    // Calculate airtime scores per band
    const airScore2g = !isNaN(max2g) ? airtimeScore(max2g, '2g') : NaN;
    const airScore5g = !isNaN(max5g) ? airtimeScore(max5g, '5g') : NaN;

    // Aggregate airtime score based on band mode
    let airtimeScoreAgg: number;
    if (bandMode === '2.4GHz') {
      airtimeScoreAgg = airScore2g;
    } else if (bandMode === '5GHz') {
      airtimeScoreAgg = airScore5g;
    } else if (bandMode === 'avg') {
      const parts: number[] = [];
      if (!isNaN(airScore2g)) parts.push(airScore2g);
      if (!isNaN(airScore5g)) parts.push(airScore5g);
      if (parts.length === 0) {
        airtimeScoreAgg = NaN;
      } else {
        const w2g = 0.6;
        const w5g = 0.4;
        const sum = (isNaN(airScore2g) ? 0 : airScore2g * w2g) +
                    (isNaN(airScore5g) ? 0 : airScore5g * w5g);
        const denom = (isNaN(airScore2g) ? 0 : w2g) + (isNaN(airScore5g) ? 0 : w5g);
        airtimeScoreAgg = denom > 0 ? sum / denom : NaN;
      }
    } else {
      // worst mode
      const scores = [airScore2g, airScore5g].filter(s => !isNaN(s));
      airtimeScoreAgg = scores.length > 0 ? Math.max(...scores) : NaN;
    }

    // No-clients relief
    let airtimeScoreAdj: number;
    if (isNaN(airtimeScoreAgg)) {
      airtimeScoreAdj = NaN;
    } else if (clientCount === 0) {
      airtimeScoreAdj = airtimeScoreAgg * 0.8; // 20% relief
    } else {
      airtimeScoreAdj = airtimeScoreAgg;
    }

    // Fill missing airtime with 0.4 (neutral)
    const airtimeScoreFilled = isNaN(airtimeScoreAdj) ? 0.4 : airtimeScoreAdj;

    // Client pressure score
    const clientScore = clientPressureScore(clientCount, p95Clients);

    // Resource health scores
    const cpuScore = cpuHealthScore(cpuUtil);
    const memUsedPct = memTotal > 0 ? (1 - memFree / memTotal) * 100 : 0;
    const memScore = memHealthScore(memUsedPct);

    // Final conflictivity (weights sum to 1.0)
    const W_AIR = 0.75;
    const W_CL = 0.15;
    const W_CPU = 0.05;
    const W_MEM = 0.05;

    const conflictivity = clamp(
      airtimeScoreFilled * W_AIR +
      clientScore * W_CL +
      cpuScore * W_CPU +
      memScore * W_MEM,
      0.0,
      1.0
    );

    // Calculate contributions for breakdown
    const breakdown = {
      airtime_contrib: airtimeScoreFilled * W_AIR,
      client_contrib: clientScore * W_CL,
      cpu_contrib: cpuScore * W_CPU,
      mem_contrib: memScore * W_MEM,
    };

    results.push({
      name,
      macaddr,
      group_code: groupCode,
      conflictivity,
      airtime_score: airtimeScoreAgg,
      airtime_score_2g: airScore2g,
      airtime_score_5g: airScore5g,
      util_2g: max2g,
      util_5g: max5g,
      client_count: clientCount,
      client_score: clientScore,
      cpu_utilization: cpuUtil,
      cpu_score: cpuScore,
      mem_used_pct: memUsedPct,
      mem_score: memScore,
      breakdown,
    });
  }

  return results;
}

function formatConflictiveAPs(
  aps: ConflictiveAP[],
  minConflictivity: number = 0.5
): string {
  // Filter and sort by conflictivity
  const conflictive = aps
    .filter(ap => ap.conflictivity >= minConflictivity)
    .sort((a, b) => b.conflictivity - a.conflictivity);

  if (conflictive.length === 0) {
    return `No hi ha punts conflictius amb conflictivitat >= ${minConflictivity.toFixed(2)}`;
  }

  let output = `=== PUNTS CONFLICTIUS (${conflictive.length} APs amb conflictivitat >= ${minConflictivity.toFixed(2)}) ===\n\n`;

  for (const ap of conflictive) {
    output += `📶 Access Point: ${ap.name}\n`;
    output += `   MAC: ${ap.macaddr}\n`;
    output += `   Edifici: ${ap.group_code}\n`;
    output += `   Conflictivitat Total: ${ap.conflictivity.toFixed(3)}\n\n`;

    output += `   Causes de la conflictivitat:\n`;
    
    // Airtime contribution
    if (ap.breakdown.airtime_contrib > 0.1) {
      output += `   • Congestió del canal (airtime): ${ap.breakdown.airtime_contrib.toFixed(3)} (${(ap.breakdown.airtime_contrib / ap.conflictivity * 100).toFixed(1)}%)\n`;
      if (!isNaN(ap.util_2g)) {
        output += `     - 2.4 GHz: ${ap.util_2g.toFixed(1)}% utilització (score: ${ap.airtime_score_2g.toFixed(3)})\n`;
      }
      if (!isNaN(ap.util_5g)) {
        output += `     - 5 GHz: ${ap.util_5g.toFixed(1)}% utilització (score: ${ap.airtime_score_5g.toFixed(3)})\n`;
      }
    }

    // Client contribution
    if (ap.breakdown.client_contrib > 0.05) {
      output += `   • Pressió de clients: ${ap.breakdown.client_contrib.toFixed(3)} (${(ap.breakdown.client_contrib / ap.conflictivity * 100).toFixed(1)}%)\n`;
      output += `     - Clients connectats: ${ap.client_count}\n`;
      output += `     - Client score: ${ap.client_score.toFixed(3)}\n`;
    }

    // CPU contribution
    if (ap.breakdown.cpu_contrib > 0.01) {
      output += `   • Ús de CPU: ${ap.breakdown.cpu_contrib.toFixed(3)} (${(ap.breakdown.cpu_contrib / ap.conflictivity * 100).toFixed(1)}%)\n`;
      output += `     - CPU utilització: ${ap.cpu_utilization.toFixed(1)}%\n`;
      output += `     - CPU score: ${ap.cpu_score.toFixed(3)}\n`;
    }

    // Memory contribution
    if (ap.breakdown.mem_contrib > 0.01) {
      output += `   • Ús de memòria: ${ap.breakdown.mem_contrib.toFixed(3)} (${(ap.breakdown.mem_contrib / ap.conflictivity * 100).toFixed(1)}%)\n`;
      output += `     - Memòria utilitzada: ${ap.mem_used_pct.toFixed(1)}%\n`;
      output += `     - Memòria score: ${ap.mem_score.toFixed(3)}\n`;
    }

    output += `\n`;
  }

  return output;
}

async function analyzeConflictivityByHour(
  hour: number,
  minConflictivity: number = 0.5,
  bandMode: BandMode = 'worst'
): Promise<string> {
  const repoRoot = process.cwd();
  const apDir = join(repoRoot, 'realData', 'ap');
  const clientsDir = join(repoRoot, 'realData', 'clients');

  // Find files for this hour
  const { apFiles, clientFiles } = await findFilesByHour(apDir, clientsDir, hour);

  if (apFiles.length === 0) {
    return `No s'han trobat arxius AP per a l'hora ${hour}:00`;
  }

  let allAPs: APRecord[] = [];

  // Read all AP files for this hour
  for (const file of apFiles) {
    const filePath = join(apDir, file);
    try {
      const apData = await readAPSnapshot(filePath, bandMode);
      allAPs = allAPs.concat(apData);
    } catch (error) {
      console.error(`Error reading ${file}:`, error);
    }
  }

  if (allAPs.length === 0) {
    return `No s'han pogut llegir dades AP per a l'hora ${hour}:00`;
  }

  // Process APs and calculate conflictivity
  const conflictiveAPs = processAPs(allAPs, bandMode);

  // Format output
  const header = `=== ANÀLISI DE CONFLICTIVITAT PER A L'HORA ${hour}:00 ===\n`;
  const info = `Arxius AP processats: ${apFiles.length}\n`;
  const info2 = `Total APs analitzats: ${allAPs.length}\n`;
  const info3 = `Mode de banda: ${bandMode}\n\n`;

  const output = header + info + info2 + info3 + formatConflictiveAPs(conflictiveAPs, minConflictivity);

  return output;
}

// Main execution
async function main() {
  const args = process.argv.slice(2);
  
  if (args.length === 0) {
    console.error('Ús: bun run conflictivity_analyzer.ts <hora> [min_conflictivity] [band_mode]');
    console.error('  hora: 0-23');
    console.error('  min_conflictivity: 0.0-1.0 (per defecte: 0.5)');
    console.error('  band_mode: worst|avg|2.4GHz|5GHz (per defecte: worst)');
    process.exit(1);
  }

  const hour = parseInt(args[0], 10);
  if (isNaN(hour) || hour < 0 || hour > 23) {
    console.error('Error: hora ha de ser un número entre 0 i 23');
    process.exit(1);
  }

  const minConflictivity = args[1] ? parseFloat(args[1]) : 0.5;
  const bandMode = (args[2] as BandMode) || 'worst';

  const result = await analyzeConflictivityByHour(hour, minConflictivity, bandMode);
  console.log(result);
}

if (import.meta.main) {
  main().catch(console.error);
}

export { analyzeConflictivityByHour, formatConflictiveAPs };

