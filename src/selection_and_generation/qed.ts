import {knex} from "knex"
const RDKit = require("rdkit")

const main = async () => {
    const db = knex({
        client: "mysql2",
        connection: {
            host: "127.0.0.1",
            port: 3306,
            user: "bio",
            password: "bio",
            database: "bio",
        }
    })

    console.log(RDKit)

    await db.destroy()
}

main().then(() => console.log("finished")).catch(console.error)