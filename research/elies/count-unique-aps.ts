import { readdir, readFile } from 'fs/promises';
import { join } from 'path';

interface APRecord {
  macaddr?: string;
}

async function countUniqueAPs() {
  const apDir = './realData/ap';
  const uniqueMacs = new Set<string>();
  const macOccurrences = new Map<string, number>();

  try {
    // Read all files in the directory
    const files = await readdir(apDir);
    const jsonFiles = files.filter(file => file.endsWith('.json'));

    console.log(`Found ${jsonFiles.length} JSON files to process...\n`);

    let totalRecords = 0;

    // Process each file
    for (const file of jsonFiles) {
      const filePath = join(apDir, file);
      try {
        const content = await readFile(filePath, 'utf-8');
        const data: APRecord[] = JSON.parse(content);

        // Track unique MAC addresses
        for (const ap of data) {
          totalRecords++;
          if (ap.macaddr) {
            uniqueMacs.add(ap.macaddr);
            macOccurrences.set(ap.macaddr, (macOccurrences.get(ap.macaddr) || 0) + 1);
          }
        }
      } catch (error) {
        console.error(`Error processing ${file}:`, error);
      }
    }

    // Display results
    console.log('=' .repeat(60));
    console.log('Unique APs Analysis (by macaddr)');
    console.log('=' .repeat(60));
    console.log(`Total AP records processed: ${totalRecords.toLocaleString()}`);
    console.log(`Unique APs (unique macaddr): ${uniqueMacs.size.toLocaleString()}`);
    console.log(`Average occurrences per AP: ${(totalRecords / uniqueMacs.size).toFixed(2)}`);
    console.log('=' .repeat(60));

    // Show some statistics about occurrences
    const occurrences = Array.from(macOccurrences.values());
    const minOccurrences = Math.min(...occurrences);
    const maxOccurrences = Math.max(...occurrences);
    const avgOccurrences = occurrences.reduce((a, b) => a + b, 0) / occurrences.length;

    console.log('\nOccurrence Statistics:');
    console.log(`  Minimum occurrences: ${minOccurrences}`);
    console.log(`  Maximum occurrences: ${maxOccurrences}`);
    console.log(`  Average occurrences: ${avgOccurrences.toFixed(2)}`);

  } catch (error) {
    console.error('Error:', error);
    process.exit(1);
  }
}

countUniqueAPs();

