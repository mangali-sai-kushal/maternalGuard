const fs = require('fs');
const { pool } = require('./src/config/db');

(async () => {
    try {
        const res = await pool.query('SELECT 1 AS ok');
        fs.writeFileSync('log.txt', 'OK: ' + JSON.stringify(res.rows));
    } catch (err) {
        fs.writeFileSync('log.txt', 'ERROR: ' + err.message + '\n' + err.stack);
    } finally {
        await pool.end();
    }
})();
