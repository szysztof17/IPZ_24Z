import {parse} from "csv-parse"
import fs from 'fs'
import {knex} from "knex";

type DBType = "string" | "float" | "int" | "id" | "longtext"

const fields: Record<string, DBType> = {
    cid: "id",
    cmpdname: "longtext",
    cmpdsynonym: "longtext",
    mw: "float",
    mf: "string",
    polararea: "float",
    complexity: "float",
    xlogp: "float",
    heavycnt: "int",
    hbonddonor: "int",
    hbondacc: "int",
    rotbonds: "int",
    inchi: "longtext",
    isosmiles: "longtext",
    canonicalsmiles: "longtext",
    inchikey: "string",
    iupacname: "longtext",
    exactmass: "float",
    monoisotopicmass: "float",
    charge: "float",
    covalentunitcnt: "float",
    isotopeatomcnt: "float",
    totalatomstereocnt: "float",
    definedatomstereocnt: "float",
    undefinedatomstereocnt: "float",
    totalbondstereocnt: "float",
    definedbondstereocnt: "float",
    undefinedbondstereocnt: "float",
    pclidcnt: "float",
    gpidcnt: "float",
    gpfamilycnt: "float",
    meshheadings: "string",
    annothits: "longtext",
    annothitcnt: "float",
    aids: "longtext",
    cidcdate: "int",
    sidsrcname: "longtext",
    depcatg: "longtext",
    annotation: "longtext",

}

const parseValue = (value: string, type: DBType) => {
    if (value === "")
        return null
    switch (type) {
        case "id":
        case "int": return parseInt(value);
        case "float": return parseFloat(value);
        case "longtext":
        case "string": return value;
    }
}

const main = async () => {
    const records = fs.createReadStream("./PubChem_compounds.csv").pipe(parse({
        columns: true
    }))

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

    await db.schema.dropTableIfExists("raw")
    await db.schema.createTable("raw", builder => {
        for (const fieldName in fields) {
            const fieldType = fields[fieldName]

            switch (fieldType) {
                case "id": builder.integer(fieldName).unique(); break
                case "int": builder.integer(fieldName).nullable(); break
                case "float": builder.float(fieldName).nullable(); break
                case "string": builder.string(fieldName).nullable(); break
                case "longtext": builder.text(fieldName).nullable(); break
            }
        }
    })

    let i = 0;

    for await (const record of records) {
        ++i
        if (i%10000 == 0)
            console.log(i)

        //fix columns
        for (const column in record) {
            const normalizedColumn = column.trim()
            if (!(normalizedColumn in fields)) throw new Error(`Unknown column "${normalizedColumn}"`)

            const value = parseValue(record[column], fields[normalizedColumn])
            delete record[column]
            record[normalizedColumn] = value
        }

        await db('raw').insert(record)
    }

    await db.destroy()
}

main().then(() => console.log("finished")).catch(console.error)