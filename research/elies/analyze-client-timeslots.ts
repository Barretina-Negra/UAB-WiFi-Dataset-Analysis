import { readdir, readFile } from 'fs/promises';
import { join } from 'path';

interface ClientRecord {
  last_connection_time?: number;
  macaddr?: string;
}

interface TimeslotStats {
  filename: string;
  timeslot: string;
  logCount: number;
  uniqueClients: number;
  minTimestamp: number;
  maxTimestamp: number;
}

function parseTimeslotFromFilename(filename: string): string | null {
  // Format: client-info-2025-04-03T00_01_15+02_00-783.json
  // Extract: 2025-04-03T00_01_15+02_00
  const match = filename.match(/client-info-(\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2}\+\d{2}_\d{2})/);
  if (match) {
    return match[1].replace(/_/g, ':');
  }
  return null;
}

function formatTimestamp(ts: number): string {
  return new Date(ts).toISOString();
}

async function analyzeClientTimeslots() {
  const clientsDir = './realData/clients';
  const timeslotStats = new Map<string, TimeslotStats>();
  let minTimestamp = Infinity;
  let maxTimestamp = -Infinity;
  const allClientMacs = new Set<string>();

  try {
    // Read all files in the directory
    const files = await readdir(clientsDir);
    const jsonFiles = files.filter(file => file.endsWith('.json'));

    console.log(`Found ${jsonFiles.length} JSON files to process...\n`);

    // Process each file
    for (const file of jsonFiles) {
      const timeslot = parseTimeslotFromFilename(file);
      if (!timeslot) {
        console.warn(`Could not parse timeslot from filename: ${file}`);
        continue;
      }

      const filePath = join(clientsDir, file);
      try {
        const content = await readFile(filePath, 'utf-8');
        const data: ClientRecord[] = JSON.parse(content);

        const uniqueClientsInFile = new Set<string>();
        let minTs = Infinity;
        let maxTs = -Infinity;

        for (const client of data) {
          if (client.macaddr) {
            uniqueClientsInFile.add(client.macaddr);
            allClientMacs.add(client.macaddr);
          }

          if (client.last_connection_time) {
            minTimestamp = Math.min(minTimestamp, client.last_connection_time);
            maxTimestamp = Math.max(maxTimestamp, client.last_connection_time);
            minTs = Math.min(minTs, client.last_connection_time);
            maxTs = Math.max(maxTs, client.last_connection_time);
          }
        }

        timeslotStats.set(timeslot, {
          filename: file,
          timeslot: timeslot,
          logCount: data.length,
          uniqueClients: uniqueClientsInFile.size,
          minTimestamp: minTs === Infinity ? 0 : minTs,
          maxTimestamp: maxTs === -Infinity ? 0 : maxTs,
        });
      } catch (error) {
        console.error(`Error processing ${file}:`, error);
      }
    }

    // Calculate date range
    if (minTimestamp === Infinity || maxTimestamp === -Infinity) {
      console.log('No timestamps found in the data.');
      return;
    }

    const dateRangeDays = Math.ceil((maxTimestamp - minTimestamp) / (1000 * 60 * 60 * 24));

    // Display results
    console.log('=' .repeat(80));
    console.log('Client Logs Analysis by Timeslot');
    console.log('=' .repeat(80));
    console.log(`\nDate Range:`);
    console.log(`  Start: ${formatTimestamp(minTimestamp)}`);
    console.log(`  End:   ${formatTimestamp(maxTimestamp)}`);
    console.log(`  Range: ${dateRangeDays} days`);
    console.log(`\nTotal unique clients across all timeslots: ${allClientMacs.size.toLocaleString()}`);
    console.log(`Total timeslots analyzed: ${timeslotStats.size}`);

    // Sort timeslots by unique clients (descending) to find peak timeslots
    const sortedTimeslots = Array.from(timeslotStats.values())
      .sort((a, b) => b.uniqueClients - a.uniqueClients);

    console.log('\n' + '=' .repeat(80));
    console.log('Top 20 Peak Timeslots (by unique clients):');
    console.log('=' .repeat(80));
    console.log('Timeslot'.padEnd(30) + 'Unique Clients'.padEnd(20) + 'Total Logs'.padEnd(15) + 'Filename');
    console.log('-'.repeat(80));

    for (let i = 0; i < Math.min(20, sortedTimeslots.length); i++) {
      const stats = sortedTimeslots[i];
      console.log(
        stats.timeslot.padEnd(30) +
        stats.uniqueClients.toLocaleString().padEnd(20) +
        stats.logCount.toLocaleString().padEnd(15) +
        stats.filename
      );
    }

    // Statistics
    const uniqueClientCounts = sortedTimeslots.map(s => s.uniqueClients);
    const logCounts = sortedTimeslots.map(s => s.logCount);

    console.log('\n' + '=' .repeat(80));
    console.log('Statistics:');
    console.log('=' .repeat(80));
    console.log(`\nUnique Clients per Timeslot:`);
    console.log(`  Min: ${Math.min(...uniqueClientCounts).toLocaleString()}`);
    console.log(`  Max: ${Math.max(...uniqueClientCounts).toLocaleString()}`);
    console.log(`  Avg: ${(uniqueClientCounts.reduce((a, b) => a + b, 0) / uniqueClientCounts.length).toFixed(2)}`);

    console.log(`\nLogs per Timeslot:`);
    console.log(`  Min: ${Math.min(...logCounts).toLocaleString()}`);
    console.log(`  Max: ${Math.max(...logCounts).toLocaleString()}`);
    console.log(`  Avg: ${(logCounts.reduce((a, b) => a + b, 0) / logCounts.length).toFixed(2)}`);

    // Find timeslots with most activity (by hour)
    const hourlyStats = new Map<number, { count: number; uniqueClients: number }>();
    for (const stats of sortedTimeslots) {
      const hour = parseInt(stats.timeslot.split('T')[1]?.split(':')[0] || '0');
      const existing = hourlyStats.get(hour) || { count: 0, uniqueClients: 0 };
      hourlyStats.set(hour, {
        count: existing.count + 1,
        uniqueClients: existing.uniqueClients + stats.uniqueClients,
      });
    }

    const sortedHours = Array.from(hourlyStats.entries())
      .sort((a, b) => b[1].uniqueClients - a[1].uniqueClients);

    console.log('\n' + '=' .repeat(80));
    console.log('Peak Hours (by average unique clients per timeslot):');
    console.log('=' .repeat(80));
    console.log('Hour'.padEnd(10) + 'Avg Unique Clients'.padEnd(25) + 'Timeslots');
    console.log('-'.repeat(80));
    for (const [hour, stats] of sortedHours.slice(0, 10)) {
      const avg = stats.uniqueClients / stats.count;
      console.log(
        `${hour.toString().padStart(2, '0')}:00`.padEnd(10) +
        avg.toFixed(2).padEnd(25) +
        stats.count.toString()
      );
    }

    console.log('\n' + '=' .repeat(80));

  } catch (error) {
    console.error('Error:', error);
    process.exit(1);
  }
}

analyzeClientTimeslots();

