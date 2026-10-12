# Deploy DataForge on Vercel

The website is a static HTML, CSS, and JavaScript project in this folder. It does not need a framework or a build step.

## Connect the repository

1. In Vercel, choose **Add New → Project** and import SelimBenAli/dataforge.
2. Set **Root Directory** to website.
3. Choose **Other** as the framework preset. Leave the build command empty and use the current directory as the output directory if Vercel asks for one.
4. Deploy. Vercel will provide a preview URL; verify the page, its styling, mobile navigation, and copy buttons there.

The repository's Python package and other files are outside the Vercel project root, so Vercel serves only this website. Once the Git repository is connected, commits to its production branch trigger new deployments.

## Connect dataforge.selimbenali.me

1. In the Vercel project, open **Settings → Domains** and add dataforge.selimbenali.me.
2. Vercel will show the DNS record required for your domain. Add that record at the service managing selimbenali.me; use the exact target Vercel displays because it can vary.
3. Wait for Vercel to verify the domain and issue HTTPS. Set dataforge.selimbenali.me as the production domain if Vercel prompts you.

The intended public address is <https://dataforge.selimbenali.me/>.
