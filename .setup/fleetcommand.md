# Domino Environment Setup in Fleetcommand

1. Go to Fleetcommand at: [https://fleetcommand.internal.domino.tech/](https://fleetcommand.internal.domino.tech/)

2. Click `Deploy` in the upper toolbar

3. Make the following changes. Leave all other entries as Default:

   - Deploy ID Prefix: `<name>`
   - Vanity URL: Enter `<customer>.domino-eval.com`
   - Flavor: Select `prod-aws-eks`
   - Domino Catalog: Select `GA`

   **Advanced Deployment Configuration**

   - Starburst: Select `Enable Starburst`
   - NetApp: Select `Enable NetApp`
   - Spot Instance: Deselect `spot instances`
   - Customer Name: Enter `<Customer Name>`
   - Cloud Billing: Select `Enable Cloud Billing`
   - Domino Governance: Select `Domino Governance`

   **Settings**

   - Deployment Scheduler: `Configure as Desired`
   - Destroy After: `Configure as desired`

4. Click `Create Deployment`
