import { readdir, readFile } from 'fs/promises';
import { join } from 'path';

interface APRecord {
  ap_deployment_mode?: string;
}

async function countDeploymentModes() {
  const apDir = './realData/ap';
  const counts = new Map<string, number>();

  try {
    // Read all files in the directory
    const files = await readdir(apDir);
    const jsonFiles = files.filter(file => file.endsWith('.json'));

    console.log(`Found ${jsonFiles.length} JSON files to process...\n`);

    // Process each file
    for (const file of jsonFiles) {
      const filePath = join(apDir, file);
      try {
        const content = await readFile(filePath, 'utf-8');
        const data: APRecord[] = JSON.parse(content);

        // Count deployment modes in this file
        for (const ap of data) {
          const mode = ap.ap_deployment_mode ?? 'null/undefined';
          counts.set(mode, (counts.get(mode) || 0) + 1);
        }
      } catch (error) {
        console.error(`Error processing ${file}:`, error);
      }
    }

    // Display results
    console.log('Unique values in "ap_deployment_mode" and their counts:');
    console.log('=' .repeat(60));

    // Sort by count (descending) for better readability
    const sortedEntries = Array.from(counts.entries())
      .sort((a, b) => b[1] - a[1]);

    for (const [mode, count] of sortedEntries) {
      console.log(`${mode.padEnd(30)} : ${count.toLocaleString()}`);
    }

    console.log('=' .repeat(60));
    console.log(`Total AP records processed: ${Array.from(counts.values()).reduce((a, b) => a + b, 0).toLocaleString()}`);
    console.log(`Unique deployment modes: ${counts.size}`);

  } catch (error) {
    console.error('Error:', error);
    process.exit(1);
  }
}

countDeploymentModes();

